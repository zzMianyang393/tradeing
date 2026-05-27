# OKX 量化自动交易系统

基于技术指标和机器学习的加密货币自动交易系统，支持 OKX 模拟盘和实盘交易。

## 功能特性

- **多时间框架分析**：15m 主时间框架 + 1h 趋势确认
- **智能信号生成**：EMA、RSI、MACD、布林带、ADX 等技术指标组合
- **机器学习辅助**：XGBoost 模型预测价格方向
- **风险控制**：动态止损、止盈、仓位管理
- **实时监控**：7×24 小时自动交易
- **回测系统**：历史数据回测验证策略

## 快速开始

### 本地运行

```bash
# 1. 克隆项目
git clone <仓库地址>
cd Quantify

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API 密钥
cp config/settings.yaml.example config/settings.yaml
# 编辑 config/settings.yaml 填入你的 OKX API 密钥

# 5. 运行回测
python main.py backtest --days 90 --symbols "BTC/USDT:USDT,ETH/USDT:USDT"

# 6. 启动模拟盘
python main.py live --symbols "BTC/USDT:USDT,ETH/USDT:USDT" --interval 60
```

### 服务器部署

详见 [DEPLOY.md](DEPLOY.md) 部署教程。

**一键部署：**
```bash
# 上传项目到服务器后
chmod +x deploy.sh
sudo ./deploy.sh
```

## 项目结构

```
Quantify/
├── config/
│   ├── settings.yaml      # API 配置和全局参数
│   └── strategy.yaml      # 策略参数
├── data/
│   ├── fetcher.py         # 数据获取
│   ├── storage.py         # 数据存储
│   └── hot_coins.py       # 热门币种筛选
├── strategy/
│   ├── indicators.py      # 技术指标计算
│   ├── signals.py         # 信号生成
│   ├── ml_model.py        # 机器学习模型
│   └── hybrid.py          # 混合策略
├── risk/
│   ├── stop_loss.py       # 止损管理
│   ├── take_profit.py     # 止盈管理
│   └── position_sizer.py  # 仓位计算
├── execution/
│   └── order_executor.py  # 订单执行
├── backtest/
│   ├── engine.py          # 回测引擎
│   ├── analyzer.py        # 结果分析
│   └── report.py          # 报告生成
├── main.py                # 主入口
├── deploy.sh              # 服务器部署脚本
├── monitor.sh             # 监控脚本
└── DEPLOY.md              # 部署文档
```

## 命令说明

```bash
# 回测
python main.py backtest --days 90 --symbols "BTC/USDT:USDT,ETH/USDT:USDT"

# 模拟盘交易
python main.py live --symbols "BTC/USDT:USDT,ETH/USDT:USDT" --interval 60

# 训练 ML 模型
python main.py train --symbol "BTC/USDT:USDT"

# 热门币种
python main.py hot --top 10

# 关闭所有持仓
python close_all.py
```

## 策略说明

### 入场条件
- **趋势确认**：EMA9 > EMA21 > EMA55（多头排列）
- **回调入场**：价格回调至 EMA9 附近（±0.5%）
- **动量确认**：RSI < 65，MACD 金叉
- **成交量**：当前成交量 > 20 日均量的 0.8 倍
- **ADX 过滤**：ADX > 15（趋势市场）

### 风险控制
- **止损**：固定 0.8%（可配置）
- **止盈**：固定 0.5%（盈亏比 1:0.625）
- **仓位**：单笔最大 10% 资金
- **杠杆**：5 倍（固定）

## 监控命令

```bash
# 健康检查
./monitor.sh check

# 查看日志
./monitor.sh log

# 查看持仓
./monitor.sh positions

# 重启服务
./monitor.sh restart
```

## 注意事项

1. **模拟盘优先**：先在模拟盘运行至少 2 周，确认策略稳定
2. **资金安全**：实盘时只投入你能承受亏损的资金
3. **API 密钥**：不要将 `settings.yaml` 提交到 Git
4. **监控告警**：建议配置 Telegram 或邮件告警
5. **定期备份**：定期备份 `data/trading.db`

## 技术栈

- Python 3.9+
- ccxt（交易所 API）
- pandas（数据处理）
- scikit-learn / xgboost（机器学习）
- loguru（日志）
- Flask（可选 Web 界面）

## 许可证

MIT License
