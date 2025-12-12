#!/bin/bash
# ============================================
# 虹膜识别服务 - 快速打包脚本（使用预编译包）
# 策略：优先使用预编译的二进制包，避免编译
# 适用：阿里云/CentOS/RHEL
# 使用方法: chmod +x package-quick.sh && ./package-quick.sh
# ============================================

set -e

echo "============================================"
echo "   虹膜识别服务打包工具 (快速版)"
echo "============================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 工作目录
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"

# 检查项目文件
if [ ! -f "iris_service.py" ]; then
    echo -e "${RED}错误: 未找到 iris_service.py${NC}"
    exit 1
fi

# ============================================
# [1/6] 检查并安装 Python 3.8+
# ============================================
echo ""
echo -e "${YELLOW}[1/6] 检查 Python 3.8+...${NC}"

# 查找可用的 Python 3.8+ 版本
PYTHON_CMD=""
for py_version in python3.11 python3.10 python3.9 python3.8; do
    if command -v $py_version &> /dev/null; then
        PYTHON_CMD=$py_version
        echo -e "${GREEN}✓ 找到 $py_version${NC}"
        break
    fi
done

# 如果没找到，尝试安装 Python 3.8
if [ -z "$PYTHON_CMD" ]; then
    echo "未找到 Python 3.8+，正在安装..."

    if command -v yum &> /dev/null; then
        # CentOS/RHEL/Alibaba Linux
        sudo yum install -y python38 python38-devel python38-pip gcc gcc-c++ 2>&1 | grep -v "already installed" || true

        if command -v python3.8 &> /dev/null; then
            PYTHON_CMD=python3.8
        fi
    elif command -v apt-get &> /dev/null; then
        # Ubuntu/Debian/Kylin
        sudo apt-get update -qq
        sudo apt-get install -y python3.8 python3.8-dev python3.8-venv python3-pip gcc g++ 2>&1 | grep -v "already installed" || true

        if command -v python3.8 &> /dev/null; then
            PYTHON_CMD=python3.8
        fi
    fi
fi

# 验证 Python 版本
if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}✗ 错误: 未找到 Python 3.8+${NC}"
    echo "请手动安装:"
    echo "  CentOS/RHEL: sudo yum install python38"
    echo "  Ubuntu/Debian: sudo apt-get install python3.8"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo -e "${GREEN}✓ 使用: $PYTHON_VERSION (命令: $PYTHON_CMD)${NC}"

# 安装系统依赖
echo ""
echo "安装系统依赖..."
if command -v yum &> /dev/null; then
    sudo yum install -y gcc gcc-c++ mesa-libGL glib2 libSM libXext libXrender 2>&1 | grep -v "already installed" || true
elif command -v apt-get &> /dev/null; then
    sudo apt-get install -y gcc g++ libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev 2>&1 | grep -v "already installed" || true
fi

echo -e "${GREEN}✓ 系统依赖已安装${NC}"

# ============================================
# [2/6] 创建虚拟环境（使用 Python 3.8+）
# ============================================
echo ""
echo -e "${YELLOW}[2/6] 创建虚拟环境...${NC}"

VENV_DIR=".venv"
rm -rf $VENV_DIR

echo "使用 $PYTHON_CMD 创建虚拟环境..."
$PYTHON_CMD -m venv $VENV_DIR

source "$VENV_DIR/bin/activate"

# 验证虚拟环境中的 Python 版本
VENV_PYTHON_VERSION=$(python --version)
echo -e "${GREEN}✓ 虚拟环境 Python: $VENV_PYTHON_VERSION${NC}"

# 升级 pip
echo "升级 pip..."
pip install --quiet --upgrade pip setuptools wheel

echo -e "${GREEN}✓ 虚拟环境已创建${NC}"

# ============================================
# [3/6] 安装 Python 依赖（使用预编译包）
# ============================================
echo ""
echo -e "${YELLOW}[3/6] 安装 Python 依赖（使用预编译包）...${NC}"

echo "配置 pip 使用预编译包..."
export PIP_PREFER_BINARY=1

echo "安装依赖（按顺序）..."

# 1. 先安装 numpy（很多包依赖它）
echo "  [1/6] 安装 numpy..."
pip install --quiet numpy>=1.20.0

# 2. 安装 OpenCV（使用预编译的二进制包）
echo "  [2/6] 安装 opencv-python（预编译版）..."
pip install --quiet opencv-python-headless>=4.5.0  # headless 版本更小，无 GUI 依赖

# 3. 安装 Flask
echo "  [3/6] 安装 flask..."
pip install --quiet flask>=2.0.0

# 4. 安装 flask-cors
echo "  [4/6] 安装 flask-cors..."
pip install --quiet flask-cors>=3.0.0

# 5. 安装 PyWavelets
echo "  [5/6] 安装 PyWavelets..."
pip install --quiet PyWavelets>=1.1.0

# 6. 安装 Pillow
echo "  [6/6] 安装 Pillow..."
pip install --quiet Pillow>=8.0.0

# 7. 安装 PyInstaller
echo "安装 PyInstaller..."
pip install --quiet pyinstaller

echo -e "${GREEN}✓ 所有依赖已安装${NC}"

# ============================================
# [4/6] 验证安装
# ============================================
echo ""
echo -e "${YELLOW}[4/6] 验证安装...${NC}"

python -c "import sys; print('Python:', sys.version)" || {
    echo -e "${RED}✗ Python 验证失败${NC}"
    exit 1
}

python -c "import cv2; print('OpenCV:', cv2.__version__)" || {
    echo -e "${RED}✗ OpenCV 导入失败${NC}"
    exit 1
}

python -c "import numpy; print('NumPy:', numpy.__version__)" || {
    echo -e "${RED}✗ NumPy 导入失败${NC}"
    exit 1
}

python -c "import flask; print('Flask:', flask.__version__)" || {
    echo -e "${RED}✗ Flask 导入失败${NC}"
    exit 1
}

echo -e "${GREEN}✓ 所有模块验证通过${NC}"

# ============================================
# [5/6] 打包应用
# ============================================
echo ""
echo -e "${YELLOW}[5/6] 打包应用...${NC}"

# 清理旧构建
rm -rf build dist *.spec

# 执行打包
echo "开始打包（预计 3-5 分钟）..."
pyinstaller \
    --onefile \
    --name iris_service \
    --strip \
    --noupx \
    --add-data "util:util" \
    --hidden-import=flask \
    --hidden-import=flask_cors \
    --hidden-import=cv2 \
    --hidden-import=numpy \
    --hidden-import=pywt \
    --hidden-import=PIL \
    --hidden-import=werkzeug \
    --collect-all cv2 \
    --collect-all numpy \
    --collect-all pywt \
    --collect-all PIL \
    --collect-all flask \
    --copy-metadata flask \
    --copy-metadata werkzeug \
    iris_service.py 2>&1 | tee pyinstaller.log

if [ ! -f "dist/iris_service" ]; then
    echo -e "${RED}✗ 打包失败，查看日志: pyinstaller.log${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 打包完成${NC}"

# ============================================
# [6/6] 创建发布包
# ============================================
echo ""
echo -e "${YELLOW}[6/6] 创建发布包...${NC}"

RELEASE_DIR="release"
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR

# 复制文件
cp dist/iris_service $RELEASE_DIR/
chmod +x $RELEASE_DIR/iris_service

mkdir -p $RELEASE_DIR/photo
mkdir -p $RELEASE_DIR/feature

# 复制数据（如果存在）
[ -d "photo" ] && [ "$(ls -A photo 2>/dev/null)" ] && cp -r photo/* $RELEASE_DIR/photo/ 2>/dev/null || true
[ -d "feature" ] && [ "$(ls -A feature 2>/dev/null)" ] && cp -r feature/* $RELEASE_DIR/feature/ 2>/dev/null || true

# 复制配置文件
[ -f "iris_service.service" ] && cp iris_service.service $RELEASE_DIR/
[ -f "deploy.sh" ] && cp deploy.sh $RELEASE_DIR/ && chmod +x $RELEASE_DIR/deploy.sh

# 打包
PACKAGE_NAME="iris_service_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf $PACKAGE_NAME -C $RELEASE_DIR .

# 完成
echo ""
echo "============================================"
echo -e "${GREEN}           打包完成！${NC}"
echo "============================================"
echo ""
echo "📦 打包文件: $PWD/$PACKAGE_NAME"
echo "📊 文件大小: $(du -h $PACKAGE_NAME | cut -f1)"
echo ""
echo "🔍 打包信息:"
echo "   Python 版本: $VENV_PYTHON_VERSION"
echo "   NumPy 版本: $(python -c 'import numpy; print(numpy.__version__)')"
echo "   OpenCV 版本: $(python -c 'import cv2; print(cv2.__version__)')"
echo ""
echo "📋 发布内容:"
ls -lh $RELEASE_DIR/
echo ""
echo "🚀 下一步操作:"
echo ""
echo "1. 传输到目标服务器:"
echo "   scp $PACKAGE_NAME root@target-server:/opt/"
echo ""
echo "2. 在目标服务器部署:"
echo "   mkdir -p /opt/iris_service"
echo "   tar -xzvf /opt/$PACKAGE_NAME -C /opt/iris_service"
echo "   cd /opt/iris_service && sudo ./deploy.sh"
echo ""
echo "3. 验证服务:"
echo "   systemctl status iris_service"
echo "   curl http://localhost:5000/api/status"
echo ""
