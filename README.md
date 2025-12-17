# 虹膜识别服务 (Iris Service)

基于 Flask 的虹膜识别 HTTP 服务，提供视频流、虹膜注册和识别功能。

## 目录结构

```
IrisService/
├── iris_service.py       # 主服务程序
├── requirements.txt      # Python 依赖
├── README.md             # 说明文档
├── util/                 # 核心算法模块
│   ├── __init__.py
│   ├── config.py         # 配置文件
│   ├── innerCircle.py    # 内圆（瞳孔）检测
│   ├── outerCircle.py    # 外圆（虹膜边界）检测
│   ├── normalize.py      # 虹膜归一化
│   ├── feature.py        # 特征提取（小波变换）
│   ├── contrast.py       # 特征对比/匹配
│   └── visualization.py  # 可视化
├── photo/                # 虹膜图像存储
└── feature/              # 特征数据存储
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 默认配置
python iris_service.py

# 指定参数
python iris_service.py --host 0.0.0.0 --port 8084 --camera 0
```

### 3. 测试服务

```bash
# 查看状态
curl http://localhost:8084/api/status

# 浏览器打开视频流
http://localhost:8084/api/video/stream
```

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 获取服务状态 |
| `/api/video/stream` | GET | MJPEG 视频流 |
| `/api/register/capture` | POST | 采集虹膜样本 |
| `/api/register/complete` | POST | 完成注册 |
| `/api/recognize` | POST | 虹膜识别 |

### 采集虹膜

```bash
curl -X POST http://localhost:8084/api/register/capture \
  -H "Content-Type: application/json" \
  -d '{"user_id": "zhangsan", "eye": "L"}'
```

### 完成注册

```bash
curl -X POST http://localhost:8084/api/register/complete \
  -H "Content-Type: application/json" \
  -d '{"user_id": "zhangsan"}'
```

### 虹膜识别

```bash
curl -X POST http://localhost:8084/api/recognize
```

## 前端集成

### 显示视频流

```html
<img src="http://localhost:8084/api/video/stream" />
```

### React 示例

```tsx
const IRIS_URL = 'http://localhost:8084';

// 视频流
<img src={`${IRIS_URL}/api/video/stream`} />

// 采集
await fetch(`${IRIS_URL}/api/register/capture`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id: 'zhangsan', eye: 'L' })
});

// 识别
const res = await fetch(`${IRIS_URL}/api/recognize`, { method: 'POST' });
const data = await res.json();
// data: { success: true, user_id: 'zhangsan', eye: 'L', confidence: 85.5 }
```

## Linux 部署

### PyInstaller 打包

```bash
pip install pyinstaller

pyinstaller --onefile \
    --add-data "util:util" \
    --hidden-import=cv2 \
    --hidden-import=numpy \
    --hidden-import=pywt \
    --name iris_service \
    iris_service.py
```

### Systemd 服务

```ini
# /etc/systemd/system/iris_service.service
[Unit]
Description=Iris Recognition Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/iris_service
ExecStart=/opt/iris_service/iris_service --host 0.0.0.0 --port 8084 --camera 0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable iris_service
sudo systemctl start iris_service
```

## 注意事项

1. **摄像头索引**：通过 `ls /dev/video*` 查看可用摄像头
2. **跨域访问**：已配置 CORS，允许所有来源
3. **防火墙**：需开放服务端口（默认 8084）

