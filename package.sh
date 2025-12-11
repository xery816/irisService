#!/bin/bash
# ============================================
# 虹膜识别服务 - 通用打包脚本
# 功能：激活虚拟环境 → 执行打包
# 适用：WSL / 任何Linux系统
# 使用方法: 
#   在项目目录下运行: ./package.sh
#   或指定目录: ./package.sh /path/to/project
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

# 确定工作目录
if [ -n "$1" ]; then
    WORK_DIR="$1"
else
    # 如果不指定，使用脚本所在目录
    WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

echo "工作目录: $WORK_DIR"

# 检查是否是项目目录
if [ ! -f "$WORK_DIR/iris_service.py" ]; then
    echo -e "${RED}错误: 未找到 iris_service.py${NC}"
    echo "请在项目目录运行此脚本，或指定项目路径："
    echo "  ./package.sh /path/to/IrisService"
    exit 1
fi

# 切换到工作目录
cd "$WORK_DIR"

# [1/4] 检查或创建虚拟环境
echo -e "${YELLOW}[1/4] 检查虚拟环境...${NC}"
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "虚拟环境不存在，正在创建..."
    python3 -m venv $VENV_DIR
    echo "✅ 虚拟环境已创建"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo "✅ 虚拟环境已激活: $PWD/$VENV_DIR"

# [2/4] 安装依赖
echo -e "${YELLOW}[2/4] 检查并安装依赖...${NC}"

# 检查requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}错误: 未找到 requirements.txt${NC}"
    exit 1
fi

# 安装项目依赖
echo "安装项目依赖..."
pip install -q -r requirements.txt

# 检查PyInstaller
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "安装 PyInstaller..."
    pip install -q pyinstaller
fi

echo "✅ 依赖检查完成"

# [3/4] 清理旧的构建文件并开始打包
echo -e "${YELLOW}[3/4] 清理旧构建文件并开始打包...${NC}"
rm -rf build dist *.spec

# 执行打包
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
    iris_service.py

# [4/4] 创建发布目录
echo -e "${YELLOW}[4/4] 创建发布包...${NC}"
RELEASE_DIR="release"
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR

# 复制文件
cp dist/iris_service $RELEASE_DIR/
cp -r photo $RELEASE_DIR/ 2>/dev/null || mkdir -p $RELEASE_DIR/photo
cp -r feature $RELEASE_DIR/ 2>/dev/null || mkdir -p $RELEASE_DIR/feature

# 复制服务文件（如果存在）
[ -f iris_service.service ] && cp iris_service.service $RELEASE_DIR/
[ -f deploy.sh ] && cp deploy.sh $RELEASE_DIR/

# 设置可执行权限
chmod +x $RELEASE_DIR/iris_service
[ -f $RELEASE_DIR/deploy.sh ] && chmod +x $RELEASE_DIR/deploy.sh

# 打包为 tar.gz
PACKAGE_NAME="iris_service_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czvf $PACKAGE_NAME -C $RELEASE_DIR .

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}           打包完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "打包文件: $PWD/$PACKAGE_NAME"
echo "文件大小: $(du -h $PACKAGE_NAME | cut -f1)"
echo ""
echo "发布内容:"
echo "  - iris_service        (可执行文件)"
echo "  - photo/              (图像目录)"
echo "  - feature/            (特征目录)"
[ -f $RELEASE_DIR/iris_service.service ] && echo "  - iris_service.service (systemd服务)"
[ -f $RELEASE_DIR/deploy.sh ] && echo "  - deploy.sh           (部署脚本)"
echo ""
echo "部署步骤:"
echo "  1. 上传到目标服务器: scp $PACKAGE_NAME root@server:/root/"
echo "  2. SSH到服务器: ssh root@server"
echo "  3. 解压: tar -xzvf $PACKAGE_NAME -C /opt/iris_service"
echo "  4. 部署: cd /opt/iris_service && ./deploy.sh"
echo ""
echo "或直接运行测试："
echo "  tar -xzvf $PACKAGE_NAME"
echo "  cd release"
echo "  ./iris_service --help"
echo ""
