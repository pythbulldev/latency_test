from typing import Any
from dataclasses import dataclass
import json

@dataclass
class BookTicker:
    e: str
    u: float
    s: str
    b: str
    B: str
    a: str
    A: str
    T: float
    E: float

    @staticmethod
    def from_dict(obj: Any) -> 'BookTicker':
        _e = str(obj.get("e"))
        _u = int(obj.get("u"))
        _s = str(obj.get("s"))
        _b = str(obj.get("b"))
        _B = str(obj.get("B"))
        _a = str(obj.get("a"))
        _A = str(obj.get("A"))
        _T = float(obj.get("T"))
        _E = float(obj.get("E"))
        return BookTicker(_e, _u, _s, _b, _B, _a, _A, _T, _E)

@dataclass
class StreamData:
    stream: str
    data: BookTicker

    @staticmethod
    def from_dict(obj: Any) -> 'StreamData':
        _stream = str(obj.get("stream"))
        _data = BookTicker.from_dict(obj.get("data"))
        return StreamData(_stream, _data)
