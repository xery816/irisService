#!/bin/bash
# ============================================
# 虹膜识别服务 - Linux 打包脚本
# 使用方法: chmod +x build.sh && ./build.sh
# ============================================

set -e

echo "============================================"
echo "       虹膜识别服务打包工具"
echo "============================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python
echo -e "${YELLOW}[1/6] 检查 Python 环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3${NC}"
    exit 1
fi
python3 --version

# 创建并激活虚拟环境
echo -e "${YELLOW}[2/6] 设置虚拟环境...${NC}"
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "创建虚拟环境..."
    python3 -m venv $VENV_DIR
fi

echo "激活虚拟环境..."
source $VENV_DIR/bin/activate

# 安装依赖
echo -e "${YELLOW}[3/6] 安装依赖...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# 检查 PyInstaller
echo -e "${YELLOW}[4/6] 检查 PyInstaller...${NC}"
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "安装 PyInstaller..."
    pip install pyinstaller
fi

# 清理旧的构建文件
echo -e "${YELLOW}[5/6] 清理旧构建文件...${NC}"
rm -rf build dist *.spec

# 执行打包
echo -e "${YELLOW}[6/6] 开始打包...${NC}"
pyinstaller \
    --onefile \
    --name iris_service \
    --add-data "util:util" \
    --hidden-import=flask \
    --hidden-import=flask_cors \
    --hidden-import=cv2 \
    --hidden-import=numpy \
    --hidden-import=pywt \
    --hidden-import=PIL \
    --collect-submodules=cv2 \
    --collect-submodules=numpy \
    --collect-submodules=pywt \
    iris_service.py

# 创建发布目录
echo -e "${YELLOW}[7/7] 创建发布包...${NC}"
RELEASE_DIR="release"
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR

# 复制文件
cp dist/iris_service $RELEASE_DIR/
cp -r photo $RELEASE_DIR/ 2>/dev/null || mkdir -p $RELEASE_DIR/photo
cp -r feature $RELEASE_DIR/ 2>/dev/null || mkdir -p $RELEASE_DIR/feature
cp iris_service.service $RELEASE_DIR/
cp deploy.sh $RELEASE_DIR/

# 设置可执行权限
chmod +x $RELEASE_DIR/iris_service
chmod +x $RELEASE_DIR/deploy.sh

# 打包为 tar.gz
PACKAGE_NAME="iris_service_$(date +%Y%m%d).tar.gz"
tar -czvf $PACKAGE_NAME -C $RELEASE_DIR .

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}           打包完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "发布目录: $RELEASE_DIR/"
echo "  - iris_service        (可执行文件)"
echo "  - photo/              (图像目录)"
echo "  - feature/            (特征目录)"
echo "  - iris_service.service (systemd服务)"
echo "  - deploy.sh           (部署脚本)"
echo ""
echo "打包文件: $PACKAGE_NAME"
echo ""
echo "部署步骤:"
echo "  1. 将 $PACKAGE_NAME 上传到目标 Linux 服务器"
echo "  2. tar -xzvf $PACKAGE_NAME -C /opt/iris_service"
echo "  3. cd /opt/iris_service && ./deploy.sh"
echo ""

