import time
from datetime import datetime

class DataPoint:
    def __init__(self, symbol: str, timestamp: datetime, open: float, high: float, low: float, close: float, volume: int) -> None:
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def __str__(self):
        return f'["symbol": "{self.symbol}", "timestamp": "{self.timestamp}", "open": "{self.open}", "high": "{self.high}", "low": "{self.low}", "close": "{self.close}", "volume": "{self.volume}"]'