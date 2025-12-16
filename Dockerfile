# ============================================
# 虹膜识别服务 Docker 镜像
# ============================================

FROM python:3.8-slim

LABEL maintainer="iris_service"
LABEL description="Iris Recognition Service with Flask and OpenCV"

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN sed -i 's@http://deb.debian.org/debian@https://mirrors.aliyun.com/debian@g' /etc/apt/sources.list \
 && sed -i 's@http://security.debian.org/debian-security@https://mirrors.aliyun.com/debian-security@g' /etc/apt/sources.list

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .

RUN pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple \
    --timeout 120 \
    -r requirements.txt


# 复制应用代码
COPY iris_service.py .
COPY util/ ./util/

# 创建数据目录
RUN mkdir -p /app/photo /app/feature

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/api/status')" || exit 1

# 启动命令
CMD ["python", "iris_service.py", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
