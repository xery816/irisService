@echo off
echo ====================================
echo     虹膜识别服务 - 启动
echo ====================================

python iris_service.py --host 0.0.0.0 --port 5000 --camera 0

pause

