#!/bin/bash
# OKX 量化交易系统 - Linux 服务器部署脚本
# 使用方法: chmod +x deploy.sh && sudo ./deploy.sh

set -e

echo "=========================================="
echo "  OKX 量化交易系统 - 服务器部署"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查是否为 root 用户
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}错误: 请使用 sudo 运行此脚本${NC}"
    echo "使用方法: sudo ./deploy.sh"
    exit 1
fi

# 获取当前用户
DEPLOY_USER=${SUDO_USER:-$USER}
DEPLOY_HOME=$(eval echo ~$DEPLOY_USER)
PROJECT_DIR="$DEPLOY_HOME/Quantify"

echo -e "${YELLOW}[1/8] 检查系统环境${NC}"

# 检测系统类型
if [ -f /etc/debian_version ]; then
    PKG_MANAGER="apt"
    echo "检测到 Debian/Ubuntu 系统"
elif [ -f /etc/redhat-release ]; then
    PKG_MANAGER="yum"
    echo "检测到 CentOS/RHEL 系统"
else
    echo -e "${RED}不支持的操作系统${NC}"
    exit 1
fi

echo -e "${YELLOW}[2/8] 更新系统${NC}"
if [ "$PKG_MANAGER" = "apt" ]; then
    apt update -y
    apt upgrade -y
    apt install -y python3 python3-pip python3-venv git
else
    yum update -y
    yum install -y python3 python3-pip git
fi

echo -e "${YELLOW}[3/8] 创建项目目录${NC}"
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
    chown $DEPLOY_USER:$DEPLOY_USER "$PROJECT_DIR"
fi

echo -e "${YELLOW}[4/8] 配置 Python 虚拟环境${NC}"
su - $DEPLOY_USER -c "
cd ~
if [ ! -d Quantify ]; then
    echo '错误: 请先将项目文件上传到 $PROJECT_DIR'
    echo '使用方法: scp -r /本地路径/Quantify/* $DEPLOY_USER@服务器IP:$PROJECT_DIR/'
    exit 1
fi
cd Quantify

# 创建虚拟环境
if [ ! -d venv ]; then
    python3 -m venv venv
fi

# 激活虚拟环境并安装依赖
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo 'Python 环境配置完成'
"

echo -e "${YELLOW}[5/8] 创建 systemd 服务${NC}"
cat > /etc/systemd/system/trading.service << EOF
[Unit]
Description=OKX Quantitative Trading System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$DEPLOY_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python main.py live --symbols "BTC/USDT:USDT,ETH/USDT:USDT" --interval 60
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/logs/trading.log
StandardError=append:$PROJECT_DIR/logs/trading.log

# 环境变量
Environment=PYTHONUNBUFFERED=1

# 安全限制
NoNewPrivileges=true

# 资源限制
MemoryMax=512M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

echo -e "${YELLOW}[6/8] 启动服务${NC}"
systemctl daemon-reload
systemctl start trading
systemctl enable trading

echo -e "${YELLOW}[7/8] 配置日志轮转${NC}"
cat > /etc/logrotate.d/trading << EOF
$PROJECT_DIR/logs/trading.log {
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

echo -e "${YELLOW}[8/8] 配置防火墙${NC}"
if command -v ufw &> /dev/null; then
    ufw allow ssh
    echo "SSH 已允许"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}  部署完成！${NC}"
echo "=========================================="
echo ""
echo "常用命令："
echo "  查看状态:    sudo systemctl status trading"
echo "  启动服务:    sudo systemctl start trading"
echo "  停止服务:    sudo systemctl stop trading"
echo "  重启服务:    sudo systemctl restart trading"
echo "  查看日志:    sudo journalctl -u trading -f"
echo "  查看交易日志: tail -f $PROJECT_DIR/logs/trading.log"
echo ""
echo "请确保已配置好 config/settings.yaml 中的 API 密钥"
echo ""
