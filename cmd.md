# 🚀 Crypto Quant Model 快捷指令指南 (cmd.md)

本文件整理了 `crypto-quant-model` 量化交易系統在**開發、回測、實盤/模擬執行**與**雲端服務維護**時所需的常用 CLI 指令。

## 🛠️ 1. 本地開發與環境啟動

### 1.1 虛擬環境與套件安裝

**Bash**

```
# 複製專案與進入目錄
git clone <repo-url>
cd crypto-quant-model

# 建立並啟用虛擬環境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows PowerShell

# 安裝依賴套件
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 1.2 連線驗證 (Binance Spot Testnet)

**Bash**

```
# 檢查 API Key 權限與 Testnet 可用 USDT 餘額
python test_conn.py
```

## 📊 2. 資料收集與特徵工程 (Data & Feature Pipeline)

**Bash**

```
# 批量下載並清洗市值前 20 大熱門幣種歷史 K 線 (1h, 15,000 筆)
python cli.py batch-collect --timeframe 1h --limit 15000

# 檢視當前生效標的池 (Active Universe) 與 0% 勝率毒瘤黑名單
python cli.py show-universe

# 批次執行所有清洗後幣種的特徵工程 (產出 Parquet 矩陣)
python cli.py batch-feature --input-dir data/interim --output-dir data/processed
```

## 🧠 3. 模型訓練與策略回測 (Train & Backtest)

### 3.1 單模型與 Walk-Forward 滾動交叉驗證

**Bash**

```
# 1. 訓練 XGBoost 二元分類器 Baseline
python cli.py train --model-name xgb_classifier --val-days 30 --horizon 12 --output models/best_xgb_model.pkl

# 2. 執行 Walk-Forward 滾動 Out-of-Sample 驗證
python cli.py wf-eval --model-name xgb_classifier --min-train-days 180 --step-days 30 --horizon 12

# 3. 執行策略歷史回測 (含手續費扣除與勝率/夏普比率計算)
python cli.py backtest --model-name xgb_classifier --prob-threshold 0.54 --fee-rate 0.0010
```

### 3.2 雙模型集成回測 (Ensemble: XGBoost + LightGBM)

**Bash**

```
# 執行 XGB+LGB 雙模型 Soft Voting 集成策略回測
python cli.py ensemble-backtest --min-train-days 180 --step-days 30 --prob-threshold 0.54
```

## ⚡ 4. 高頻 SL/TP 網格尋優與診斷報表 (Phase 2 & Phase 3 Core)

### 4.1 1m 高頻 K 線 SL/TP 網格碰撞尋優 (`sltp_backtest.py`)

> **開發規範**：累積 50+ 筆無 SL/TP 限制的 Baseline 數據後，帶入此腳本計算最佳 TP/SL 甜點區（如 TP 2.5% / SL 4.0%），並產出 Heatmap 與 Equity Curve^^。

**Bash**

```
# 執行 1m K 線逐分碰撞模擬，產出網格熱力圖與最優風控組合
python cli.py backtest-sltp
# 或直接透過模組執行
python -m scripts.sltp_backtest
```

### 4.2 機構級 2x2 模型診斷報表 (`analyst_log.py`)

**Bash**

```
# 生成包含機率分佈、勝率矩陣、價格關聯之暗黑主題診斷圖表 (diagnostic_report.png)
python cli.py diag --input logs/inference_history.csv --trades logs/paper_trades.csv
# 或直接透過模組執行
python -m scripts.analyst_log
```

## 🚀 5. 實盤 / 模擬交易執行 (Paper & Live Trading)

### 5.1 Ensemble Spot Live Trader (Testnet + Native OCO)

**Bash**

```
# 1. 純預測 Dry-Run 模式（不下單，僅寫入 Log 與 Telegram 推播）
python -m src.live.live_trader_ensemble --dry-run

# 2. 單次執行推論與開倉測試（方便除錯）
python -m src.live.live_trader_ensemble --once

# 3. 啟動常駐排程（整點 00 分 01 秒自動對齊觸發）
python -m src.live.live_trader_ensemble
```

### 5.2 Binance Testnet 帳戶資產重置 (清空持倉與掛單)

**Bash**

```
# 執行安全重置腳本：撤銷所有 Open Orders / OCO，並將非 USDT 資產市價清空
python -m scripts.reset_binance_testnet
```

## 🖥️ 6. 遠端雲端伺服器維護 (SSH & Systemd Service)

### 6.1 SSH 連線與伺服器管理

**Bash**

```
# 連線至遠端 EC2 / 雲端伺服器
ssh -i ".ssh/Personal.pem" ubuntu@ec2-16-176-146-106.ap-southeast-2.compute.amazonaws.com

# 進入專案目錄並啟用虛擬環境
cd /home/ubuntu/crypto-quant-model
source .venv/bin/activate
```

### 6.2 Systemd 服務設定範本 (`/etc/systemd/system/quant-trader.service`)

在伺服器上記錄並維護實盤交易腳本為背景 Daemon 服務：

**Ini, TOML**

```
[Unit]
Description=Crypto Quant Model Ensemble Live Trader Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/crypto-quant-model
ExecStart=/home/ubuntu/crypto-quant-model/.venv/bin/python -m src.live.live_trader_ensemble
Restart=always
RestartSec=10
EnvironmentFile=/home/ubuntu/crypto-quant-model/.env

[Install]
WantedBy=multi-user.target
```

### 6.3 Systemd 常用管理命令

**Bash**

```
# 1. 重新載入服務設定檔
sudo systemctl daemon-reload

# 2. 啟動 / 停止 / 重啟量化交易服務
sudo systemctl start quant-trader
sudo systemctl stop quant-trader
sudo systemctl restart quant-trader

# 3. 設定開機自動啟動
sudo systemctl enable quant-trader

# 4. 檢查服務當前運行狀態
sudo systemctl status quant-trader

# 5. 實時查看交易系統執行 Log (最後 100 行並持續監控)
journalctl -u quant-trader -n 100 -f
```
