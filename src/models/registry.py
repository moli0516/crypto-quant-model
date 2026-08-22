import logging
from typing import Dict, Type, Any
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    模型工廠與註冊器 (Model Registry / Factory)
    用於動態註冊、管理與創建不同的機器學習模型包裝類別。
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        類別裝飾器：用於將模型類別註冊到工廠中
        
        用法範例:
            @ModelRegistry.register("xgb_classifier")
            class XGBClassifierWrapper(BaseModel):
                ...
        """
        def decorator(subclass: Type[BaseModel]):
            if not issubclass(subclass, BaseModel):
                raise TypeError(f"【錯誤】被註冊的類別 '{subclass.__name__}' 必須繼承自 BaseModel！")
            
            if name in cls._registry:
                logger.warning(f"⚠️ 警告: 模型名稱 '{name}' 已存在於註冊表中，將會被覆蓋。")
                
            cls._registry[name] = subclass
            logger.info(f"📌 成功註冊模型: '{name}' -> {subclass.__name__}")
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, model_params: Any = None) -> BaseModel:
        """
        根據模型名稱動態創建模型實例
        
        :param name: 模型註冊名稱 (如 "xgb_classifier")
        :param model_params: 傳入模型的超參數字典
        :return: 對應模型的實例 (BaseModel 的子類別)
        """
        if name not in cls._registry:
            available_models = list(cls._registry.keys())
            raise ValueError(f"【錯誤】找不到名為 '{name}' 的模型！現有可用的模型列表為: {available_models}")
        
        model_cls = cls._registry[name]
        logger.info(f"🔨 正在創建模型實例: '{name}' ({model_cls.__name__})")
        
        if model_params is not None:
            return model_cls(model_params=model_params)
        return model_cls()

    @classmethod
    def list_models(cls) -> list:
        """列出目前所有已註冊的模型名稱"""
        return list(cls._registry.keys())