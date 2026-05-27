# OKX 量化交易系统 - 服务器部署教程

## 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 1核 | 2核 |
| 内存 | 1GB | 2GB |
| 磁盘 | 10GB | 20GB |
| 系统 | Ubuntu 20.04 / CentOS 7 | Ubuntu 22.04 |
| Python | 3.9+ | 3.11 |

---

## 第一步：准备服务器

### 1.1 SSH 登录服务器

```bash
ssh root@你的服务器IP
```

### 1.2 安装 Python 和依赖

```bash
# Ubuntu/Debian
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git

# CentOS/RHEL
yum update -y
yum install -y python3 python3-pip git
```

### 1.3 创建专用用户（推荐）

```bash
# 创建 trading 用户
useradd -m -s /bin/bash trading
su - trading
```

---

## 第二步：上传项目文件

### 方式 A：使用 Git（推荐）

```bash
cd /home/trading
git clone <你的仓库地址> Quantify
cd Quantify
```

### 方式 B：手动上传

在本地电脑执行：
```bash
# Windows (PowerShell)
scp -r E:\ai-trade\Quantify\* root@服务器IP:/home/trading/Quantify/

# Linux/Mac
scp -r ~/ai-trade/Quantify/* root@服务器IP:/home/trading/Quantify/
```

---

## 第三步：配置项目

### 3.1 创建虚拟环境

```bash
cd /home/trading/Quantify
python3 -m venv venv
source venv/bin/activate
```

### 3.2 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 配置 API 密钥

```bash
# 编辑配置文件
nano config/settings.yaml
```

修改为你的 OKX API 密钥：
```yaml
okx:
  api_key: "你的API_KEY"
  secret_key: "你的SECRET_KEY"
  passphrase: "你的PASSPHRASE"
  sandbox: true  # 模拟盘保持 true
```

### 3.4 测试连接

```bash
python main.py live --symbols "BTC/USDT:USDT" --interval 120
```

确认看到 "连接成功" 和 "余额" 信息后，按 Ctrl+C 停止。

---

## 第四步：配置 Systemd 服务（推荐）

### 4.1 创建服务文件

```bash
sudo nano /etc/systemd/system/trading.service
```

粘贴以下内容：

```ini
[Unit]
Description=OKX Quantitative Trading System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=trading
WorkingDirectory=/home/trading/Quantify
ExecStart=/home/trading/Quantify/venv/bin/python main.py live --symbols "BTC/USDT:USDT,ETH/USDT:USDT" --interval 60
Restart=always
RestartSec=30
StandardOutput=append:/home/trading/Quantify/logs/trading.log
StandardError=append:/home/trading/Quantify/logs/trading.log

# 环境变量
Environment=PYTHONUNBUFFERED=1

# 安全限制
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/home/trading/Quantify/data /home/trading/Quantify/logs

# 资源限制
MemoryMax=512M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
```

### 4.2 启动服务

```bash
# 重新加载配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start trading

# 设置开机自启
sudo systemctl enable trading

# 查看状态
sudo systemctl status trading
```

### 4.3 常用管理命令

```bash
# 查看实时日志
sudo journalctl -u trading -f

# 停止服务
sudo systemctl stop trading

# 重启服务
sudo systemctl restart trading

# 查看服务状态
sudo systemctl status trading

# 查看最近日志
sudo journalctl -u trading --since "1 hour ago"
```

---

## 第五步：配置日志轮转

### 5.1 使用 logrotate（推荐）

```bash
sudo nano /etc/logrotate.d/trading
```

粘贴以下内容：

```
/home/trading/Quantify/logs/trading.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    maxsize 50M
}
```

### 5.2 测试配置

```bash
sudo logrotate -d /etc/logrotate.d/trading
```

---

## 第六步：监控和维护

### 6.1 设置监控脚本

创建监控脚本：
```bash
nano /home/trading/Quantify/monitor.sh
```

```bash
#!/bin/bash
# 监控脚本 - 检查交易系统是否正常运行

SERVICE="trading"
LOG_FILE="/home/trading/Quantify/logs/trading.log"
ALERT_EMAIL="your-email@example.com"  # 可选：配置邮件通知

# 检查服务状态
if ! systemctl is-active --quiet $SERVICE; then
    echo "$(date): 服务 $SERVICE 已停止，正在重启..." >> /home/trading/Quantify/logs/monitor.log
    sudo systemctl restart $SERVICE
fi

# 检查日志文件大小（超过 100MB 告警）
LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null)
if [ "$LOG_SIZE" -gt 104857600 ]; then
    echo "$(date): 警告 - 日志文件超过 100MB" >> /home/trading/Quantify/logs/monitor.log
fi
```

设置执行权限：
```bash
chmod +x /home/trading/Quantify/monitor.sh
```

### 6.2 添加定时任务

```bash
crontab -e
```

添加以下行：
```bash
# 每 5 分钟检查一次服务状态
*/5 * * * * /home/trading/Quantify/monitor.sh

# 每天凌晨 3 点清理旧日志
0 3 * * * find /home/trading/Quantify/logs -name "*.log.gz" -mtime +30 -delete
```

### 6.3 查看交易状态

```bash
# 查看最新日志
tail -f /home/trading/Quantify/logs/trading.log

# 查看今日交易记录
grep "$(date +%Y-%m-%d)" /home/trading/Quantify/logs/trading.log | grep -E "\[开仓\]|\[平仓\]|\[止盈\]|\[止损\]"

# 查看持仓情况
grep "\[Tick\]" /home/trading/Quantify/logs/trading.log | tail -10
```

---

## 第七步：安全配置（重要）

### 7.1 配置防火墙

```bash
# Ubuntu (ufw)
sudo ufw allow ssh
sudo ufw enable

# CentOS (firewalld)
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload
```

### 7.2 禁用 root 登录（可选但推荐）

```bash
sudo nano /etc/ssh/sshd_config
```

修改：
```
PermitRootLogin no
```

重启 SSH：
```bash
sudo systemctl restart sshd
```

### 7.3 设置 SSH 密钥登录

```bash
# 在本地生成密钥
ssh-keygen -t ed25519

# 上传公钥到服务器
ssh-copy-id trading@服务器IP
```

---

## 故障排查

### 问题 1：服务无法启动

```bash
# 查看详细错误
sudo journalctl -u trading -n 50

# 检查 Python 环境
cd /home/trading/Quantify
source venv/bin/activate
python main.py --help
```

### 问题 2：API 连接失败

```bash
# 测试网络连接
ping www.okx.com

# 检查 API 配置
python -c "
import ccxt
exchange = ccxt.okx({'enableRateLimit': True})
print(exchange.fetch_ticker('BTC/USDT:USDT'))
"
```

### 问题 3：日志文件过大

```bash
# 手动清理
sudo truncate -s 0 /home/trading/Quantify/logs/trading.log

# 检查 logrotate 状态
sudo logrotate -d /etc/logrotate.d/trading
```

### 问题 4：磁盘空间不足

```bash
# 查看磁盘使用
df -h

# 清理旧日志
find /home/trading/Quantify/logs -name "*.log.gz" -mtime +7 -delete

# 清理数据库缓存
find /home/trading/Quantify/data -name "*.db" -exec sqlite3 {} "VACUUM;" \;
```

---

## 部署检查清单

- [ ] 服务器系统更新完成
- [ ] Python 3.9+ 安装完成
- [ ] 项目文件上传完成
- [ ] 虚拟环境创建完成
- [ ] 依赖安装完成
- [ ] API 密钥配置完成
- [ ] 连接测试通过
- [ ] Systemd 服务配置完成
- [ ] 服务启动并设置开机自启
- [ ] 日志轮转配置完成
- [ ] 防火墙配置完成
- [ ] 监控脚本配置完成

---

## 快速部署命令（一键脚本）

如果你不想手动操作，可以使用以下一键部署脚本：

```bash
#!/bin/bash
# quick_deploy.sh - 一键部署脚本

set -e

echo "=== OKX 量化交易系统 - 一键部署 ==="

# 1. 系统更新
echo "[1/7] 更新系统..."
apt update && apt upgrade -y

# 2. 安装依赖
echo "[2/7] 安装 Python 和依赖..."
apt install -y python3 python3-pip python3-venv git

# 3. 创建用户
echo "[3/7] 创建 trading 用户..."
useradd -m -s /bin/bash trading || true

# 4. 配置项目
echo "[4/7] 配置项目..."
su - trading -c "
cd ~
if [ ! -d Quantify ]; then
    echo '请先上传项目文件到 /home/trading/Quantify'
    exit 1
fi
cd Quantify
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
"

# 5. 创建 systemd 服务
echo "[5/7] 配置系统服务..."
cat > /etc/systemd/system/trading.service << 'EOF'
[Unit]
Description=OKX Quantitative Trading System
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/home/trading/Quantify
ExecStart=/home/trading/Quantify/venv/bin/python main.py live --symbols "BTC/USDT:USDT,ETH/USDT:USDT" --interval 60
Restart=always
RestartSec=30
StandardOutput=append:/home/trading/Quantify/logs/trading.log
StandardError=append:/home/trading/Quantify/logs/trading.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# 6. 启动服务
echo "[6/7] 启动服务..."
systemctl daemon-reload
systemctl start trading
systemctl enable trading

# 7. 配置日志轮转
echo "[7/7] 配置日志轮转..."
cat > /etc/logrotate.d/trading << 'EOF'
/home/trading/Quantify/logs/trading.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    maxsize 50M
}
EOF

echo "=== 部署完成 ==="
echo "查看状态: sudo systemctl status trading"
echo "查看日志: sudo journalctl -u trading -f"
```

使用方法：
```bash
chmod +x quick_deploy.sh
sudo ./quick_deploy.sh
```

---

## 注意事项

1. **API 密钥安全**：不要将 `settings.yaml` 提交到 Git 仓库
2. **模拟盘优先**：先在模拟盘运行至少 2 周，确认策略稳定后再考虑实盘
3. **资金安全**：实盘时只投入你能承受亏损的资金
4. **监控告警**：建议配置 Telegram 或邮件告警，及时发现异常
5. **定期备份**：定期备份 `data/trading.db` 数据库文件

---

## 联系支持

如遇问题，请提供以下信息：
1. 服务器系统版本：`uname -a`
2. Python 版本：`python3 --version`
3. 错误日志：`sudo journalctl -u trading -n 100`
4. 服务状态：`sudo systemctl status trading`
