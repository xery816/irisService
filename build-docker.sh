#!/bin/bash
# ============================================
# 在阿里云服务器构建 Docker 镜像
# 使用方法: chmod +x build-docker.sh && ./build-docker.sh
# ============================================

set -e

echo "=========================================="
echo "  虹膜识别服务 - Docker 镜像构建"
echo "=========================================="
echo ""

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# [1/5] 检查 Docker
echo -e "${YELLOW}[1/5] 检查 Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker 未安装，正在安装...${NC}"
    curl -fsSL https://get.docker.com | sh
    systemctl start docker
    systemctl enable docker
    echo -e "${GREEN}✓ Docker 已安装${NC}"
else
    echo -e "${GREEN}✓ Docker 已安装: $(docker --version)${NC}"
fi

# [2/5] 检查必需文件
echo ""
echo -e "${YELLOW}[2/5] 检查必需文件...${NC}"
REQUIRED_FILES=("Dockerfile" "iris_service.py" "requirements.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ 缺少文件: $file${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ $file${NC}"
done

if [ ! -d "util" ]; then
    echo -e "${RED}✗ 缺少目录: util/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ util/${NC}"

# [3/5] 构建镜像
echo ""
echo -e "${YELLOW}[3/5] 构建 Docker 镜像...${NC}"
echo "这可能需要 5-10 分钟（首次构建会下载基础镜像）"
echo ""

docker build -t iris_service:latest .

echo ""
echo -e "${GREEN}✓ 镜像构建完成${NC}"

# [4/5] 导出镜像
echo ""
echo -e "${YELLOW}[4/5] 导出镜像...${NC}"
docker save -o iris_service.tar iris_service:latest
echo -e "${GREEN}✓ 已导出: iris_service.tar${NC}"

# [5/5] 压缩
echo ""
echo -e "${YELLOW}[5/5] 压缩镜像...${NC}"
gzip -f iris_service.tar
echo -e "${GREEN}✓ 已压缩: iris_service.tar.gz${NC}"

# 完成
echo ""
echo "=========================================="
echo -e "${GREEN}       构建完成！${NC}"
echo "=========================================="
echo ""
echo "镜像文件: $(pwd)/iris_service.tar.gz"
echo "文件大小: $(du -h iris_service.tar.gz | cut -f1)"
echo ""
echo "下一步 - 传输到目标机器:"
echo "  scp iris_service.tar.gz root@target-server:/opt/"
echo ""
echo "或在目标机器上部署:"
echo "  1. 解压: gunzip iris_service.tar.gz"
echo "  2. 导入: docker load -i iris_service.tar"
echo "  3. 运行: docker run -d --name iris_service \\"
echo "             --device=/dev/video0:/dev/video0 \\"
echo "             -p 5000:5000 \\"
echo "             -v /opt/iris_service/photo:/app/photo \\"
echo "             -v /opt/iris_service/feature:/app/feature \\"
echo "             --restart=unless-stopped \\"
echo "             iris_service:latest"
echo ""
