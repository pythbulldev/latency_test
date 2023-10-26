import threading
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import stats
from scipy.stats import kstest

from connection.binance_connection import SimpleBinanceConnector
from data_model.binance_messages import StreamData

symbols = ['btcusdt']
SOCKETS_NUMBER = 5
MAX_INTERVALS = 5
TIME_INTERVAL = 60

def do_latency_tests_and_calculation(data: pd.DataFrame, interval_count: int):
    df = data.copy()

    df_sorted = df.sort_values(by=['Book_Ticker_ID', 'Book_Ticker_Receive_Time'])
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted['Latency'] = df_sorted['Book_Ticker_Receive_Time'] - df_sorted['Book_Ticker_Server_Time']
    df_sorted['Socket_ID'] = df_sorted['Socket_ID'].astype(int)

    fast_updates_hyst = [0] * SOCKETS_NUMBER
    latency_data_by_socket_lists = [[] for _ in range(SOCKETS_NUMBER)]
    book_ticker_id = 0
    latency_data_buf = []
    corrupt_data_frames = 0
    corrupted_data_frames_ids = []

    for index in range(len(df_sorted)):
        row = df_sorted.iloc[index]
        if(row['Book_Ticker_ID'] == book_ticker_id):
            latency_data_buf.append(row)
        else:
            #We need to filter out all book_ticker_ids that were not reccived from all SOCKETS_NUMBER 
            #Otherwise we will exclude some part of big latencies and corrupt statistic. That is why we use data only if it satisfies condition: 
            if(len(latency_data_buf) == SOCKETS_NUMBER):
                index_h = int(latency_data_buf[0]['Socket_ID'])
                fast_updates_hyst[index_h] += 1
                for buf in latency_data_buf:
                    socket_id = int(buf['Socket_ID'])
                    latency = buf['Latency']
                    latency_data_by_socket_lists[socket_id].append(latency)
            #And we keep track of volume of corrupted data
            else:
                corrupt_data_frames += 1
                corrupted_data_frames_ids.append(book_ticker_id)

            latency_data_buf.clear()
            latency_data_buf.append(row)
            book_ticker_id = row['Book_Ticker_ID']

    #Looking for min latency and shifting data on it's value
    mins = [np.min(_) for _ in latency_data_by_socket_lists]
    min_latency = np.min(mins)
    for i in range(SOCKETS_NUMBER):
        latency_data_by_socket_lists[i] = latency_data_by_socket_lists[i] - min_latency

    ltest_matrix, vars = run_ltest_for_data(latency_data_by_socket_lists)
    ttest_matrix, means = run_ttest_for_data(latency_data_by_socket_lists, ltest_matrix)
    
    #Adding exponential data sample to compare
    mean = np.mean(latency_data_by_socket_lists[0])
    size = len(latency_data_by_socket_lists[0])
    etalon = np.random.exponential(mean, size=size)
    latency_data_by_socket_lists.append(etalon)

    run_ks_tests_for_data(latency_data_by_socket_lists)
    save_cdf_fu_qq_charts(latency_data_by_socket_lists, fast_updates_hyst, interval_count)
    
    print("T-test matrix:")
    print(ttest_matrix)
    print("Latency means: ")
    print(means)

    print("Levene test matrix:")
    print(ltest_matrix)
    print("Latency vars: ")
    print(vars)
    
    print(fast_updates_hyst)
    print(corrupt_data_frames)

    return fast_updates_hyst

def save_cdf_fu_qq_charts(data_lists, fast_updates, interval_count):

    for i in range(SOCKETS_NUMBER + 1):
        #Creating probability distribultion functions
        unique_values, counts = np.unique(data_lists[i], return_counts=True)
        cumulative_counts = np.cumsum(counts)
        cumulative_probabilities = cumulative_counts / len(data_lists[i])
        sorted_unique_values = np.sort(unique_values)
        label = f"Socket {i} latency dist" if i < SOCKETS_NUMBER else f"Ideal exponential dist"
        plt.step(sorted_unique_values, cumulative_probabilities, where='mid', label=label)
        
    plt.xlabel('Value')
    plt.ylabel('Cumulative probability')
    plt.title('Empiric distribution function')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname = f"Empiric distribution function {interval_count}", dpi=2000)
    plt.clf()
 
    for i in range(SOCKETS_NUMBER + 1):
        stats.probplot(data_lists[i], dist = 'expon', plot=pylab)
        filename = f"Socket {i} QQ chart (expon) {interval_count}" if i < SOCKETS_NUMBER else f"Ideal exponential dist QQ chart {interval_count}"
        pylab.savefig(filename)
        pylab.clf()

    updates_sum = sum(fast_updates)
    fast_updates_pers = [stat/updates_sum for stat in fast_updates]
    plt.scatter(np.arange(0, SOCKETS_NUMBER), fast_updates_pers)
    plt.title('Fast updates')
    plt.savefig(fname = f"Fast updates {interval_count}", dpi=1000)
    plt.clf()

def run_regression(Arg:list, Func:list, count: int):
        X = np.array(Arg)
        Y = np.array(Func)
        a, b, r, p, std = stats.linregress(X, Y)
        Y_pred = a * X + b
        reg_results = f"a: {a}; b: {b}; r^2: {r**2}; p-val: {p}"
        plt.scatter(X, Y, label='Data')
        plt.plot(X, Y_pred, color='red')
        plt.title(f'Linear regression {count}: {reg_results}', fontsize = 5)
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.savefig(fname = f"Linear regression {count}", dpi=2000)
        plt.clf()
        print(reg_results)

def run_ks_tests_for_data(data_series):
    count = 1
    for data in data_series:
        mean =  np.mean(data)
        ks_statistic, ks_p_value = kstest(data, 'expon', args=(0, mean))
        alpha = 0.05
        if ks_p_value > alpha:
            print(f"Data in soket {count} can have exponential DF, p-value: {ks_p_value}")
        else:
            print(f"Data in soket {count} can NOT have exponential DF, p-value: {ks_p_value}")
        count += 1

def run_ttest_for_data(data_series, var_matrix):
    n = len(data_series)
    ttest_matrix = np.zeros((n, n))
    means = [np.mean(data) for data in data_series]
    alpha = 0.01
    for i in range(n):
        for j in range(i, n):
            if i!=j:
                stat, p_value = stats.ttest_ind(data_series[i], data_series[j], equal_var = True if var_matrix[i,j] == 1 else False)
                if p_value < alpha:
                    ttest_matrix[i,j] = 0
                    ttest_matrix[j,i] = 0
                else:
                    ttest_matrix[i,j] = 1
                    ttest_matrix[j,i] = 1
            else:
                ttest_matrix[j,i] = 1

    return ttest_matrix, means

def run_ltest_for_data(data_series):
    n = len(data_series)
    ltest_matrix = np.zeros((n, n))
    vars = [np.var(data) for data in data_series]
    alpha = 0.01
    for i in range(n):
        for j in range(i, n):
            if i!=j:
                stat, p_value = stats.levene(data_series[i], data_series[j])
                if p_value < alpha:
                    ltest_matrix[i,j] = 0
                    ltest_matrix[j,i] = 0
                else:
                    ltest_matrix[i,j] = 1
                    ltest_matrix[j,i] = 1
            else:
                ltest_matrix[j,i] = 1

    return ltest_matrix, vars

if __name__ == "__main__":
    connectors = []
    opened_sockets_count = 0 
    socket_opening_order = [0] * SOCKETS_NUMBER

    latency_list = []
    column_names = ['Socket_ID', 'Book_Ticker_ID', 'Book_Ticker_Server_Time', 'Book_Ticker_Receive_Time']
    conector_message_locker = threading.Lock()

    is_new_data_for_processing = False
    latency_df = pd.DataFrame()
    first_update_cumm_stat = [0] * SOCKETS_NUMBER

    start_time = time.time() * 1000
    end_time = time.time() * 1000 + 1000 * TIME_INTERVAL 

    interval_count = 0

    for i in range(SOCKETS_NUMBER):
        id = i
        connector = SimpleBinanceConnector(id)
        connectors.append(connector)

        def on_message(ws, message, id = i):
            reccive_time = time.time()*1000
            
            if(opened_sockets_count != SOCKETS_NUMBER):
                return
            
            jsonstring = json.loads(message)
            book_ticker = StreamData.from_dict(jsonstring).data
            new_row = {'Socket_ID': id, 'Book_Ticker_ID': book_ticker.u, 'Book_Ticker_Server_Time': book_ticker.T, 'Book_Ticker_Receive_Time': reccive_time}
            
            global end_time
            with conector_message_locker:
                if(reccive_time > end_time):
                    global is_new_data_for_processing 
                    global latency_df
                    global start_time
                    
                    global interval_count
                    latency_df = pd.DataFrame(latency_list, columns = column_names)
                    latency_list.clear()
                    start_time = time.time()*1000
                    end_time = time.time()*1000 + 1000 * TIME_INTERVAL
                    is_new_data_for_processing = True 
                
                if(interval_count > MAX_INTERVALS):
                    for connector in connectors:
                        connector.unsubscribe_quotes()
                else:        
                    latency_list.append(new_row)

        def on_error(ws, error, id = i):
            print(f"Error in {id}: {error}")

        def on_close(ws, close_status_code, close_msg, id = i):
            with conector_message_locker:
                print(f"Closed {id}")

        def on_open(ws, id = i):
            with conector_message_locker:
                global opened_sockets_count
                opened_sockets_count += 1
                socket_opening_order[id] = opened_sockets_count
                if(opened_sockets_count == SOCKETS_NUMBER):
                    print(f"Opened order: {socket_opening_order}")
                    global  start_time, end_time
                    start_time = time.time() * 1000
                    end_time = time.time() * 1000 + 1000 * TIME_INTERVAL 

        connector.subscribe_quotes(symbols, on_open, on_message, on_error, on_close)

    # The stats in this application will be calculated once or in t < 1min, so there is no need to 
    # realise a cuncurrent calculation
    while True:
        if(is_new_data_for_processing):
            first_updates_stat = do_latency_tests_and_calculation(latency_df, interval_count)
            for i in range(SOCKETS_NUMBER):
                first_update_cumm_stat[i] += first_updates_stat[i]
            
            updates_sum = sum(first_update_cumm_stat)
            run_regression(socket_opening_order, [stat / updates_sum for stat in first_update_cumm_stat], interval_count)
            is_new_data_for_processing = False
            interval_count +=1
        else:    
            time.sleep(1)