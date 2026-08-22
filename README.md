# crypto-quant-model

加密貨幣量化交易模型專案。

## 專案簡介

本專案提供完整的量化交易研究與執行框架，包含資料收集、清洗、特徵工程、模型訓練與交易執行模組，並內建防止資料洩漏（Data Leakage）機制。

## 安裝步驟

```bash
# 1. Clone 專案
git clone <repo-url>
cd crypto-quant-model

# 2. 建立虛擬環境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 安裝依賴
pip install -r requirements.txt

# 開發環境額外安裝
pip install -r requirements-dev.txt
```

## CLI 用法

```bash
python cli.py --help
```

## 目錄結構

詳見專案根目錄檔案樹與 `docs/architecture.md`。

## License

MIT
