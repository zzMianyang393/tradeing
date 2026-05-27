#!/bin/bash
# 交易系统监控脚本
# 用法: ./monitor.sh [check|log|status|positions]

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$PROJECT_DIR/logs/trading.log"
SERVICE_NAME="trading"

case "${1:-check}" in
    check)
        echo "=== 交易系统健康检查 ==="
        echo ""

        # 检查服务状态
        if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
            echo "✓ 服务状态: 运行中"
        else
            echo "✗ 服务状态: 已停止"
            echo "  尝试重启: sudo systemctl restart $SERVICE_NAME"
        fi

        # 检查日志文件
        if [ -f "$LOG_FILE" ]; then
            LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
            LOG_LINES=$(wc -l < "$LOG_FILE")
            echo "✓ 日志文件: $LOG_SIZE ($LOG_LINES 行)"

            # 检查最近活动
            LAST_ACTIVITY=$(tail -1 "$LOG_FILE" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1)
            if [ -n "$LAST_ACTIVITY" ]; then
                echo "✓ 最后活动: $LAST_ACTIVITY"
            fi
        else
            echo "✗ 日志文件: 不存在"
        fi

        # 检查磁盘空间
        DISK_USAGE=$(df -h "$PROJECT_DIR" | awk 'NR==2 {print $5}' | tr -d '%')
        if [ "$DISK_USAGE" -lt 80 ]; then
            echo "✓ 磁盘使用: ${DISK_USAGE}%"
        else
            echo "⚠ 磁盘使用: ${DISK_USAGE}% (建议清理)"
        fi

        # 检查数据库
        DB_FILE="$PROJECT_DIR/data/trading.db"
        if [ -f "$DB_FILE" ]; then
            DB_SIZE=$(du -h "$DB_FILE" | cut -f1)
            echo "✓ 数据库: $DB_SIZE"
        fi

        echo ""
        ;;

    log)
        echo "=== 最近交易日志 ==="
        echo ""
        if [ -f "$LOG_FILE" ]; then
            # 显示最近的交易活动
            echo "最近 10 条 Tick:"
            grep "\[Tick\]" "$LOG_FILE" | tail -10
            echo ""
            echo "最近 5 条交易信号:"
            grep -E "\[开仓\]|\[平仓\]|\[止盈\]|\[止损\]" "$LOG_FILE" | tail -5
        else
            echo "日志文件不存在"
        fi
        echo ""
        ;;

    status)
        echo "=== 服务详细状态 ==="
        echo ""
        sudo systemctl status $SERVICE_NAME --no-pager
        echo ""
        ;;

    positions)
        echo "=== 当前持仓 ==="
        echo ""
        if [ -f "$LOG_FILE" ]; then
            # 获取最新的 Tick 信息
            LATEST_TICK=$(grep "\[Tick\]" "$LOG_FILE" | tail -1)
            if [ -n "$LATEST_TICK" ]; then
                echo "最新状态: $LATEST_TICK"
            fi

            # 获取今日交易
            TODAY=$(date +%Y-%m-%d)
            echo ""
            echo "今日交易:"
            grep "$TODAY" "$LOG_FILE" | grep -E "\[开仓\]|\[平仓\]" | head -20
        else
            echo "日志文件不存在"
        fi
        echo ""
        ;;

    restart)
        echo "=== 重启交易服务 ==="
        echo ""
        sudo systemctl restart $SERVICE_NAME
        sleep 2
        if systemctl is-active --quiet $SERVICE_NAME; then
            echo "✓ 服务重启成功"
        else
            echo "✗ 服务重启失败"
            sudo journalctl -u $SERVICE_NAME -n 10 --no-pager
        fi
        echo ""
        ;;

    *)
        echo "用法: $0 [check|log|status|positions|restart]"
        echo ""
        echo "命令说明:"
        echo "  check     - 健康检查（默认）"
        echo "  log       - 查看最近交易日志"
        echo "  status    - 查看服务详细状态"
        echo "  positions - 查看当前持仓"
        echo "  restart   - 重启服务"
        echo ""
        ;;
esac
