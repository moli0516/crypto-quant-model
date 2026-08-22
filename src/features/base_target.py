import abc
import pandas as pd

class BaseTarget(abc.ABC):
    """
    模型目標變數 (Target) 的抽象基底類別。
    """

    @abc.abstractmethod
    def generate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [介面合約] 生成機器學習所需的標籤。
        """
        pass