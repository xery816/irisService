#!/bin/bash
# ============================================
# 在目标机器部署 Docker 镜像
# 使用方法:
#   1. 上传 iris_service.tar.gz 到目标机器
#   2. chmod +x deploy-docker.sh && ./deploy-docker.sh
# ============================================

set -e

IMAGE_FILE="iris_service.tar.gz"
INSTALL_DIR="/opt/iris_service"
CONTAINER_NAME="iris_service"

echo "=========================================="
echo "  虹膜识别服务 - Docker 部署"
echo "=========================================="
echo ""

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# [1/6] 检查摄像头设备
echo -e "${YELLOW}[1/6] 检查摄像头设备...${NC}"
echo ""

# 列出当前可用的摄像头设备
if ls /dev/video* >/dev/null 2>&1; then
    DEVICES=($(ls /dev/video* 2>/dev/null | sort))

    if [ ${#DEVICES[@]} -eq 0 ]; then
        echo -e "${RED}✗ 未找到任何摄像头设备${NC}"
        echo -e "${YELLOW}⚠ 警告: 将在没有摄像头的情况下启动容器${NC}"
    else
        echo -e "${GREEN}✓ 发现 ${#DEVICES[@]} 个摄像头设备:${NC}"
        for device in "${DEVICES[@]}"; do
            # 尝试获取设备信息
            if [ -c "$device" ]; then
                device_info=$(v4l2-ctl --device=$device --info 2>/dev/null | grep "Card type" | cut -d: -f2 | xargs || echo "未知设备")
                echo "    $device - $device_info"
            else
                echo "    $device"
            fi
        done
        echo -e "${GREEN}✓ 所有设备将自动映射到容器${NC}"
    fi
else
    echo -e "${RED}✗ 未找到任何摄像头设备${NC}"
    echo -e "${YELLOW}⚠ 警告: 将在没有摄像头的情况下启动容器${NC}"
fi
echo ""

# [2/6] 检查镜像文件
echo -e "${YELLOW}[2/6] 检查镜像文件...${NC}"
if [ ! -f "$IMAGE_FILE" ]; then
    echo -e "${RED}✗ 未找到镜像文件: $IMAGE_FILE${NC}"
    echo "请先从阿里云服务器传输文件："
    echo "  scp root@aliyun-server:/path/iris_service.tar.gz ./"
    exit 1
fi
echo -e "${GREEN}✓ 找到镜像文件: $(du -h $IMAGE_FILE | cut -f1)${NC}"

# [3/6] 检查 Docker
echo ""
echo -e "${YELLOW}[3/6] 检查 Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker 未安装，正在安装...${NC}"
    curl -fsSL https://get.docker.com | sh
    systemctl start docker
    systemctl enable docker
    echo -e "${GREEN}✓ Docker 已安装${NC}"
else
    echo -e "${GREEN}✓ Docker 已安装: $(docker --version)${NC}"
fi

# [4/6] 导入镜像
echo ""
echo -e "${YELLOW}[4/6] 导入镜像...${NC}"

# 解压
if [[ $IMAGE_FILE == *.gz ]]; then
    echo "解压镜像..."
    gunzip -c $IMAGE_FILE > "${IMAGE_FILE%.gz}"  # 使用 -c 保留原始文件
    IMAGE_FILE="${IMAGE_FILE%.gz}"
fi

# 导入
echo "导入到 Docker..."
docker load -i $IMAGE_FILE

echo -e "${GREEN}✓ 镜像已导入${NC}"

# 验证
docker images | grep iris_service

# [5/6] 创建目录
echo ""
echo -e "${YELLOW}[5/6] 创建数据目录...${NC}"
mkdir -p $INSTALL_DIR/photo
mkdir -p $INSTALL_DIR/feature
echo -e "${GREEN}✓ 目录已创建: $INSTALL_DIR${NC}"

# [6/6] 停止旧容器并启动新容器
echo ""
echo -e "${YELLOW}[6/6] 停止旧容器并启动新容器...${NC}"
if docker ps -a | grep -q $CONTAINER_NAME; then
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo -e "${GREEN}✓ 旧容器已停止${NC}"
else
    echo "无旧容器"
fi

echo ""
# 映射所有 /dev/video* 设备到容器
DEVICE_FLAGS=""
echo "正在映射所有摄像头设备..."
for dev in /dev/video*; do
    if [ -c "$dev" ]; then
        DEVICE_FLAGS="$DEVICE_FLAGS --device=$dev:$dev"
        echo -e "${GREEN}✓ 映射: $dev${NC}"
    fi
done

if [ -z "$DEVICE_FLAGS" ]; then
    echo -e "${YELLOW}⚠ 警告: 未找到任何可用的摄像头设备${NC}"
    echo -e "${YELLOW}⚠ 容器将在没有摄像头的情况下启动${NC}"
fi

echo ""

# 运行容器
docker run -d \
  --name $CONTAINER_NAME \
  $DEVICE_FLAGS \
  -p 8084:8084 \
  -v $INSTALL_DIR/photo:/app/photo \
  -v $INSTALL_DIR/feature:/app/feature \
  --restart=unless-stopped \
  iris_service:latest

# 等待启动
echo "等待服务启动..."
sleep 3

# 检查状态
echo ""
if docker ps | grep -q $CONTAINER_NAME; then
    echo "=========================================="
    echo -e "${GREEN}       部署成功！${NC}"
    echo "=========================================="
    echo ""

    # 获取 IP
    IP=$(hostname -I | awk '{print $1}' || echo "localhost")

    echo "服务地址:"
    echo -e "  ${GREEN}状态:   http://$IP:8084/api/status${NC}"
    echo -e "  ${GREEN}视频流: http://$IP:8084/api/video/stream${NC}"
    echo ""

    echo "数据目录:"
    echo "  照片: $INSTALL_DIR/photo"
    echo "  特征: $INSTALL_DIR/feature"
    echo ""

    echo "常用命令:"
    echo "  docker logs $CONTAINER_NAME -f        # 查看日志"
    echo "  docker restart $CONTAINER_NAME        # 重启服务"
    echo "  docker stop $CONTAINER_NAME           # 停止服务"
    echo "  docker exec -it $CONTAINER_NAME bash  # 进入容器"
    echo ""

    # 测试 API
    echo "测试 API..."
    sleep 2
    if curl -s http://localhost:8084/api/status > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API 响应正常${NC}"
    else
        echo -e "${YELLOW}⚠ API 暂未响应，查看日志: docker logs $CONTAINER_NAME${NC}"
    fi

else
    echo "=========================================="
    echo -e "${RED}       部署失败${NC}"
    echo "=========================================="
    echo ""
    echo "查看日志："
    docker logs $CONTAINER_NAME
    exit 1
fi
