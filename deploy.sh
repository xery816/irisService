#!/bin/bash
# ============================================
# 虹膜识别服务 - Linux 部署脚本
# 使用方法: chmod +x deploy.sh && sudo ./deploy.sh
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="/opt/iris_service"
SERVICE_NAME="iris_service"

echo "============================================"
echo "       虹膜识别服务部署工具"
echo "============================================"

# 检查 root 权限
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}请使用 sudo 运行此脚本${NC}"
    exit 1
fi

# 停止已有服务
echo -e "${YELLOW}[1/6] 停止已有服务...${NC}"
systemctl stop $SERVICE_NAME 2>/dev/null || true
systemctl disable $SERVICE_NAME 2>/dev/null || true

# 创建安装目录
echo -e "${YELLOW}[2/6] 创建安装目录...${NC}"
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/photo
mkdir -p $INSTALL_DIR/feature

# 复制文件
echo -e "${YELLOW}[3/6] 复制文件...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/iris_service" ]; then
    # 检查源和目标是否相同，避免重复复制
    if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
        cp "$SCRIPT_DIR/iris_service" $INSTALL_DIR/
        chmod +x $INSTALL_DIR/iris_service
        echo "已复制 iris_service 到 $INSTALL_DIR"
    else
        echo "已在安装目录，跳过复制 iris_service"
        chmod +x $INSTALL_DIR/iris_service
    fi
else
    echo -e "${RED}错误: 未找到 iris_service 可执行文件${NC}"
    exit 1
fi

# 保留已有的数据
if [ -d "$SCRIPT_DIR/photo" ] && [ "$(ls -A $SCRIPT_DIR/photo 2>/dev/null)" ]; then
    cp -r "$SCRIPT_DIR/photo/"* $INSTALL_DIR/photo/ 2>/dev/null || true
fi

if [ -d "$SCRIPT_DIR/feature" ] && [ "$(ls -A $SCRIPT_DIR/feature 2>/dev/null)" ]; then
    cp -r "$SCRIPT_DIR/feature/"* $INSTALL_DIR/feature/ 2>/dev/null || true
fi

# 安装 systemd 服务
echo -e "${YELLOW}[4/6] 安装 systemd 服务...${NC}"
cp "$SCRIPT_DIR/iris_service.service" /etc/systemd/system/
systemctl daemon-reload

# 配置防火墙
echo -e "${YELLOW}[5/6] 配置防火墙...${NC}"
if command -v firewall-cmd &> /dev/null; then
    firewall-cmd --permanent --add-port=5000/tcp 2>/dev/null || true
    firewall-cmd --reload 2>/dev/null || true
    echo "firewalld: 端口 5000 已开放"
elif command -v ufw &> /dev/null; then
    ufw allow 5000/tcp 2>/dev/null || true
    echo "ufw: 端口 5000 已开放"
else
    echo "提示: 请手动开放防火墙端口 5000"
fi

# 启动服务
echo -e "${YELLOW}[6/6] 启动服务...${NC}"
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME

# 等待启动
sleep 2

# 检查状态
echo ""
echo "============================================"
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}       部署成功！服务已启动${NC}"
else
    echo -e "${RED}       服务启动失败${NC}"
    echo "查看日志: journalctl -u $SERVICE_NAME -f"
fi
echo "============================================"
echo ""
echo "安装目录: $INSTALL_DIR"
echo ""
echo "常用命令:"
echo "  systemctl status $SERVICE_NAME   # 查看状态"
echo "  systemctl restart $SERVICE_NAME  # 重启服务"
echo "  systemctl stop $SERVICE_NAME     # 停止服务"
echo "  journalctl -u $SERVICE_NAME -f   # 查看日志"
echo ""
echo "服务地址:"
echo "  状态: http://$(hostname -I | awk '{print $1}'):5000/api/status"
echo "  视频流: http://$(hostname -I | awk '{print $1}'):5000/api/video/stream"
echo ""

