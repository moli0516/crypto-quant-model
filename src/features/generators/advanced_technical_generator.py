import numpy as np
import pandas as pd

class AdvancedTechnicalFeatureGenerator:
    """
    計算進階技術指標（RSI, MACD, Bollinger Bands 寬度）的特徵生成器。
    執行順序安排在基礎生成器之後 (EXECUTION_ORDER = 110)
    """
    EXECUTION_ORDER = 110

    def __init__(self, **kwargs) -> None:
        pass

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        close = df["close"]

        # 1. 14 小時相對強弱指標 (RSI)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        # 使用 Wilder's smoothing (Exponential Moving Average)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        feat["rsi_14h"] = 100.0 - (100.0 / (1.0 + rs))

        # 2. MACD (預設 12, 26, 9)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        feat["macd"] = macd_line
        feat["macd_signal"] = signal_line
        feat["macd_hist"] = macd_line - signal_line

        # 3. Bollinger Bands 寬度 (20 小時, 2 個標準差)
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper_band = sma20 + (2.0 * std20)
        lower_band = sma20 - (2.0 * std20)
        
        # 布林帶寬度 (Bandwidth) = (上軌 - 下軌) / 中軌
        feat["bb_width_20h"] = (upper_band - lower_band) / (sma20 + 1e-10)
        
        # 價格在布林帶中的相對位置 (Percent B)
        feat["bb_pct_b_20h"] = (close - lower_band) / (upper_band - lower_band + 1e-10)

        return feat