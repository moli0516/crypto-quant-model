import importlib
import inspect
import os
import pkgutil

def load_all_generators(**kwargs):
    """
    動態掃描並載入當前目錄下所有的 Generator 類別，
    自動讀取 class 內部的 EXECUTION_ORDER 進行排序後回傳實例列表。
    """
    generator_instances = []

    pkg_dir = os.path.dirname(__file__)
    pkg_name = __name__

    for _, module_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
        if module_name.startswith("_") or is_pkg:
            continue

        full_module_name = f"{pkg_name}.{module_name}"
        module = importlib.import_module(full_module_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == full_module_name
                and name.endswith("Generator")
            ):
                # 傳入 kwargs 彈性初始化生成器
                generator_instances.append(obj(**kwargs))

    # 依照各 Generator 類別內部的 EXECUTION_ORDER 屬性進行排序 (未定義者預設值為 500)
    generator_instances.sort(
        key=lambda gen: getattr(gen, "EXECUTION_ORDER", 500)
    )

    return generator_instances