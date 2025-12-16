# -*- coding: utf-8 -*-
"""
虹膜识别 HTTP 服务1
启动命令: python iris_service.py --host 0.0.0.0 --port 5000 --camera 0
"""

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import threading
import time
import os
import shutil
import numpy as np

app = Flask(__name__)
CORS(app, origins="*")


class IrisService:
    def __init__(self):
        self.camera_index = 0  # 固定使用索引 0
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.is_iris_detected = False
        self.detection_result = {'detected': False}
        self.lock = threading.Lock()

    def start_camera(self):
        """启动红外摄像头"""
        import time
        total_start = time.time()
        
        print(f"[1/3] 正在打开摄像头 {self.camera_index}...")
        step_start = time.time()
        
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            step1_time = time.time() - step_start
            print(f"耗时: {step1_time:.3f}秒")
            
            if not self.cap.isOpened():
                print(f"摄像头 {self.camera_index} 打开失败！")
                return False
            
            # 不设置分辨率，使用默认值，避免5秒延迟
            # 读取一帧测试并获取默认分辨率
            print(f"[2/3] 测试读取第一帧（使用默认分辨率）...")
            step_start = time.time()
            ret, test_frame = self.cap.read()
            step2_time = time.time() - step_start
            print(f"耗时: {step2_time:.3f}秒")
            
            if ret:
                print(f"第一帧读取成功，默认分辨率: {test_frame.shape}")
            else:
                print(f"第一帧读取失败")
        
        print(f"[3/3] 启动采集线程...")
        step_start = time.time()
        self.is_running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        step3_time = time.time() - step_start
        print(f"耗时: {step3_time:.3f}秒")
        
        total_time = time.time() - total_start
        print(f"摄像头 {self.camera_index} 启动完成")
        print(f"总耗时: {total_time:.3f}秒")
        return True

    def stop_camera(self):
        """停止摄像头"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("摄像头已停止")

    def _capture_loop(self):
        """摄像头采集循环"""
        frame_count = 0
        error_count = 0
        
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame_count += 1
                    if frame_count == 1:
                        print(f"采集线程：已采集第一帧")
                    elif frame_count % 30 == 0:
                        print(f"采集线程：已采集 {frame_count} 帧")
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    with self.lock:
                        self.current_frame = gray
                    self._detect_iris(gray)
                    error_count = 0  # 重置错误计数
                else:
                    error_count += 1
                    if error_count == 1:
                        print(f"采集线程：读取帧失败")
                    if error_count >= 10:
                        print(f"采集线程：连续失败 {error_count} 次，可能摄像头异常")
            else:
                print(f"采集线程：摄像头未打开或已关闭")
                break
            
            time.sleep(0.03)  # ~30fps
        
        print(f"采集线程已停止，共采集 {frame_count} 帧")

    def _detect_iris(self, frame):
        """虹膜检测"""
        try:
            from util.innerCircle import innerCircle
            from util.outerCircle import outerCircle

            inner = innerCircle(frame)
            outer = outerCircle(frame, inner)

            self.is_iris_detected = True
            self.detection_result = {
                'detected': True,
                'inner': inner.tolist(),
                'outer': outer.tolist()
            }
        except Exception:
            self.is_iris_detected = False
            self.detection_result = {'detected': False}

    def get_video_frame(self):
        """获取带检测标记的视频帧"""
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()

        # 绘制检测圆
        if self.is_iris_detected and self.detection_result.get('detected'):
            from util.visualization import displayCircle
            inner = self.detection_result['inner']
            outer = self.detection_result['outer']
            frame = displayCircle(frame,
                                  inner[0], inner[1], inner[2],
                                  outer[0], outer[1], outer[2])
        return frame

    def capture_sample(self, user_id, eye):
        """采集一张虹膜样本"""
        if not self.is_iris_detected:
            return {'success': False, 'error': '未检测到虹膜'}

        with self.lock:
            if self.current_frame is None:
                return {'success': False, 'error': '无法获取图像'}
            frame = self.current_frame.copy()

        # 创建用户目录
        user_dir = os.path.join('photo', user_id, eye)

        # 🆕 如果是该用户该眼别的第一张照片（idx=1），清空旧数据
        if os.path.exists(user_dir):
            existing = [f for f in os.listdir(user_dir) if f.endswith('.jpeg')]
            if len(existing) == 0:
                # 目录存在但为空，说明是新的采集
                pass
            elif len(existing) > 0:
                # 检查是否是第一次采集本次注册（通过检查索引判断）
                # 如果目录下已有3张照片，说明上次注册完成了，这次是重新注册
                if len(existing) >= 3:
                    print(f"[清理旧数据] 删除 {user_id}/{eye} 的旧照片：{len(existing)} 张")
                    shutil.rmtree(user_dir)

        os.makedirs(user_dir, exist_ok=True)

        # 计算当前索引
        existing = len([f for f in os.listdir(user_dir) if f.endswith('.jpeg')])
        idx = existing + 1
        timestamp = int(time.time())
        filename = f'{timestamp}_{eye}_{idx}.jpeg'
        photo_path = os.path.join(user_dir, filename)

        # 保存图像
        cv2.imwrite(photo_path, frame)

        if not os.path.exists(photo_path):
            return {'success': False, 'error': '保存失败'}

        print(f"采集成功: {photo_path}")
        return {
            'success': True,
            'path': photo_path,
            'eye': eye,
            'index': idx
        }

    def generate_features(self, user_id=None):
        """生成特征数据集"""
        try:
            # 🆕 如果指定了 user_id，先删除该用户的旧特征
            if user_id:
                feature_user_dir = os.path.join('feature', user_id)
                if os.path.exists(feature_user_dir):
                    print(f"[清理旧特征] 删除 {user_id} 的旧特征数据")
                    shutil.rmtree(feature_user_dir)

            from util.feature import generateFeatureDataset
            print("开始生成特征数据集...")
            generateFeatureDataset()
            print("特征数据集生成完成")
            return {'success': True}
        except Exception as e:
            print(f"特征生成失败: {e}")
            return {'success': False, 'error': str(e)}

    def recognize(self):
        """虹膜识别"""
        if not self.is_iris_detected:
            return {'success': False, 'error': '未检测到虹膜'}

        with self.lock:
            if self.current_frame is None:
                return {'success': False, 'error': '无法获取图像'}
            frame = self.current_frame.copy()

        try:
            from util.contrast import contrast
            name, eye, scores = contrast(frame)

            if name and scores:
                best_score = scores[0][1]
                confidence = max(0, 100 - (best_score / 1000))
                print(f"识别成功: {name}, 眼别: {eye}, 置信度: {confidence:.2f}%")
                return {
                    'success': True,
                    'user_id': name,
                    'eye': eye,
                    'confidence': round(confidence, 2),
                    'score': best_score
                }

            return {'success': False, 'error': '未找到匹配'}
        except Exception as e:
            print(f"识别失败: {e}")
            return {'success': False, 'error': str(e)}


# 全局服务实例
iris_service = IrisService()


# ==================== HTTP API ====================

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取服务状态"""
    return jsonify({
        'service_running': True,
        'camera_running': iris_service.is_running,
        'camera_index': 0  # 固定返回 0
    })


@app.route('/api/camera/list', methods=['GET'])
def list_cameras():
    """列出所有可用摄像头"""
    cameras = []
    debug_info = []  # 调试信息

    # 1. 获取所有 /dev/video* 设备路径（使用 glob，更可靠）
    import glob
    device_paths = {}
    try:
        video_devices = glob.glob('/dev/video*')
        for device_path in video_devices:
            # 从路径中提取索引号，例如 /dev/video0 -> 0
            device_name = device_path.split('/')[-1]  # video0
            if device_name[5:].isdigit():  # 确保 video 后面是数字
                idx = int(device_name.replace('video', ''))
                device_paths[idx] = device_path
    except Exception as e:
        print(f"[摄像头列表] 读取 /dev/video* 设备失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

    print(f"[摄像头列表] 发现 {len(device_paths)} 个设备节点: {list(device_paths.values())}")

    # 临时抑制 OpenCV 警告日志
    import os
    original_log_level = os.environ.get('OPENCV_LOG_LEVEL')
    os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'  # 只显示致命错误

    try:
        # 2. 使用 OpenCV 测试哪些索引可用
        for idx in sorted(device_paths.keys()):
            device_path = device_paths[idx]
            debug_entry = {
                'index': idx,
                'device': device_path,
                'opencv_opened': False,
                'error': None,
                'device_caps': None
            }

            try:
                # 先检查设备能力
                try:
                    import subprocess
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', device_path, '--all'],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )

                    if result.returncode == 0:
                        # 提取设备能力
                        for line in result.stdout.split('\n'):
                            if 'Device Caps' in line:
                                debug_entry['device_caps'] = line.strip()
                            if 'Card type' in line:
                                debug_entry['card_type'] = line.split(':', 1)[1].strip()
                except Exception as e:
                    debug_entry['v4l2_error'] = str(e)

                # 尝试用 OpenCV 打开
                cap = cv2.VideoCapture(idx)

                if cap.isOpened():
                    debug_entry['opencv_opened'] = True

                    # 获取 OpenCV 能提供的信息
                    camera_info = {
                        'index': idx,
                        'device': device_path,
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'name': debug_entry.get('card_type', f'Camera {idx}')
                    }

                    cameras.append(camera_info)
                    cap.release()
                    print(f"[摄像头列表] ✓ {device_path} 可用: {camera_info['name']}")
                else:
                    debug_entry['error'] = 'OpenCV failed to open (可能是元数据设备)'
                    print(f"[摄像头列表] ✗ {device_path} 无法打开 (可能是元数据设备)")

            except Exception as e:
                debug_entry['error'] = str(e)
                print(f"[摄像头列表] ✗ {device_path} 异常: {e}")

            debug_info.append(debug_entry)

    finally:
        # 恢复原始日志级别
        if original_log_level is not None:
            os.environ['OPENCV_LOG_LEVEL'] = original_log_level
        else:
            os.environ.pop('OPENCV_LOG_LEVEL', None)

    return jsonify({
        'success': True,
        'cameras': cameras,
        'debug': debug_info  # 返回调试信息
    })


@app.route('/api/camera/set-device', methods=['POST'])
def set_camera_device():
    """设置摄像头设备（支持 /dev/videoX 或索引号）"""
    data = request.json or {}
    device = data.get('device')

    if not device:
        return jsonify({'success': False, 'error': '缺少 device 参数'})

    # 解析设备路径或索引
    if isinstance(device, str) and device.startswith('/dev/video'):
        # 如果是设备路径，提取索引号
        try:
            index = int(device.replace('/dev/video', ''))
        except ValueError:
            return jsonify({'success': False, 'error': f'无效的设备路径: {device}'})
    elif isinstance(device, (int, str)):
        # 如果是索引号
        try:
            index = int(device)
        except ValueError:
            return jsonify({'success': False, 'error': 'device 必须是数字或 /dev/videoX 格式'})
    else:
        return jsonify({'success': False, 'error': 'device 格式错误'})

    if index < 0:
        return jsonify({'success': False, 'error': '索引必须大于等于0'})

    # 验证设备是否存在
    device_path = f'/dev/video{index}'
    if not os.path.exists(device_path):
        return jsonify({'success': False, 'error': f'设备 {device_path} 不存在'})

    # 如果摄像头正在运行，先停止再切换
    was_running = iris_service.is_running
    if was_running:
        print(f"[设置设备] 停止当前摄像头 (索引 {iris_service.camera_index})")
        iris_service.stop_camera()

    # 设置新索引
    iris_service.camera_index = index
    print(f"[设置设备] 摄像头已设置为 {device_path} (索引 {index})")

    # 如果之前在运行，重新启动
    if was_running:
        print(f"[设置设备] 重新启动摄像头 {device_path}")
        iris_service.start_camera()
        return jsonify({
            'success': True,
            'message': f'摄像头已设置为 {device_path} 并重新启动',
            'device': device_path,
            'index': index
        })

    return jsonify({
        'success': True,
        'message': f'摄像头已设置为 {device_path}',
        'device': device_path,
        'index': index
    })


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """启动摄像头"""
    if iris_service.is_running:
        return jsonify({'success': False, 'error': '摄像头已在运行'})

    try:
        iris_service.start_camera()
        return jsonify({'success': True, 'message': '摄像头已启动'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera_api():
    """停止摄像头"""
    if not iris_service.is_running:
        return jsonify({'success': False, 'error': '摄像头未运行'})
    
    try:
        iris_service.stop_camera()
        return jsonify({'success': True, 'message': '摄像头已停止'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/stream')
def video_stream():
    """MJPEG 视频流 - 异步启动摄像头"""
    # 异步启动摄像头，不阻塞HTTP响应
    if not iris_service.is_running:
        print("[视频流] 异步启动摄像头...")
        threading.Thread(target=iris_service.start_camera, daemon=True).start()
    
    def generate():
        # 等待摄像头启动并采集到第一帧
        timeout = 10  # 10秒超时
        start_time = time.time()
        
        while iris_service.get_video_frame() is None:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                # 超时，返回错误提示
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, 'Camera Timeout', (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                return
            
            # 显示加载中提示
            loading_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            loading_text = f'Loading... {int(elapsed)}s'
            cv2.putText(loading_frame, loading_text, (180, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            _, buffer = cv2.imencode('.jpg', loading_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.3)  # 每0.3秒更新一次
        
        # 摄像头已就绪，开始推送真实视频流
        print("[视频流] 摄像头已就绪，开始推送视频")
        while True:
            frame = iris_service.get_video_frame()
            if frame is not None:
                # 确保是 BGR 格式用于编码
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30fps

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/register/capture', methods=['POST'])
def capture_iris():
    """采集虹膜样本"""
    data = request.json or {}
    user_id = data.get('user_id')
    eye = data.get('eye', 'L')

    if not user_id:
        return jsonify({'success': False, 'error': '缺少 user_id'})
    if eye not in ('L', 'R'):
        return jsonify({'success': False, 'error': 'eye 必须是 L 或 R'})

    result = iris_service.capture_sample(user_id, eye)
    return jsonify(result)


@app.route('/api/register/complete', methods=['POST'])
def complete_registration():
    """完成注册，生成特征 - 完成后自动关闭摄像头"""
    data = request.json or {}
    user_id = data.get('user_id')
    result = iris_service.generate_features(user_id)
    
    # 注册完成后自动关闭摄像头
    if iris_service.is_running:
        print("[注册完成] 自动关闭摄像头...")
        iris_service.stop_camera()
    
    return jsonify(result)


@app.route('/api/recognize', methods=['POST'])
def recognize_iris():
    """虹膜识别 - 自动启动和关闭摄像头"""
    # 识别前自动启动摄像头
    camera_was_running = iris_service.is_running
    if not camera_was_running:
        print("[虹膜识别] 自动启动摄像头...")
        iris_service.start_camera()
        time.sleep(0.5)  # 等待摄像头启动并采集第一帧
    
    # 执行识别
    result = iris_service.recognize()
    
    # 如果是临时启动的，识别后自动关闭
    if not camera_was_running:
        print("[虹膜识别] 自动关闭摄像头...")
        iris_service.stop_camera()
    
    return jsonify(result)


# ==================== 启动 ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='虹膜识别服务')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='监听端口 (默认: 5000)')
    args = parser.parse_args()

    # 摄像头索引固定为 0
    print("=" * 50)
    print(f"虹膜识别服务启动")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"摄像头索引: 0 (未启动)")
    print("")
    print("API 端点:")
    print(f"  - 状态查询: http://{args.host}:{args.port}/api/status")
    print(f"  - 启动摄像头: POST http://{args.host}:{args.port}/api/camera/start")
    print(f"  - 停止摄像头: POST http://{args.host}:{args.port}/api/camera/stop")
    print(f"  - 视频流: http://{args.host}:{args.port}/api/video/stream")
    print("")
    print("提示: 摄像头需要手动启动后才能使用")
    print("=" * 50)

    app.run(host=args.host, port=args.port, threaded=True)

