import websocket
import threading

class SimpleBinanceConnector:
    def __init__(self, id) -> None:
        self.id = id
        self._subscribed_quotes_symbols: list
        self._subscribed = False
        self._websocket_endpoint = 'wss://fstream.binance.com'
        self._ws: websocket.WebSocketApp
        self._socket_thread: threading.Thread
        self._on_open_callback: function
        self._on_message_callback: function
        self._on_error_callback: function
        self._on_close_callback: function

    def subscribe_quotes(self, symbols: list, on_open_callback, on_message_callback, on_error_callback, on_close_callback):
        if(self._subscribed):
            return
        
        self._subscribed = True
        self._subscribed_quotes_symbols = symbols.copy()
        self._on_open_callback = on_open_callback
        self._on_message_callback = on_message_callback
        self._on_error_callback = on_error_callback
        self._on_close_callback = on_close_callback

        streams = [f"{symbol}@bookTicker" for symbol in symbols]
        stream_string = "/".join(streams)
        websocket_url = f"{self._websocket_endpoint}/stream?streams={stream_string}"

        self._ws = websocket.WebSocketApp(websocket_url,
                                on_message = self._on_message_callback,
                                on_error = self._on_error_callback,
                                on_close = self._on_close_callback)

        self._ws.on_open = self._on_open_callback
        self._socket_thread = threading.Thread(target = self._ws.run_forever)
        self._socket_thread.daemon = True
        self._socket_thread.start()

    def unsubscribe_quotes(self):
        self._subscribed = False
        self._ws.close()



