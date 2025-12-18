# -*- coding: utf-8 -*-
"""
虹膜识别 HTTP 服务1
启动命令: python iris_service.py --host 0.0.0.0 --port 8084 --camera 0
"""

from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
import cv2
import threading
import time
import os
import shutil
import numpy as np
import zipfile
import requests
import re

app = Flask(__name__)
CORS(app, origins="*")


def score_ir_device(v4l2_output: str) -> int:
    """
    根据 v4l2-ctl --all 输出给设备打一个"红外相机可能性分数"

    评分维度：
    1. 名字/卡类型里包含 nir/ir/infrared/mono 等 (+3分/关键词)
    2. 像素格式中出现经典灰度格式 Y8/GREY/Y16 等 (+2分/格式)
    3. 是否有 mono 描述 (+2分)
    4. 只有一个 Video Capture 节点并且是灰度 (+1分)

    返回：
        int: IR 得分，>=3 分认为是红外摄像头
    """
    text = v4l2_output.lower()
    score = 0

    # 1. 名字/卡类型里包含 nir/ir/infrared/mono 等
    name_keywords = ['nir', 'infrared', 'ir camera', 'ir_cam', 'monochrome', 'mono']
    for kw in name_keywords:
        if kw in text:
            score += 3
            print(f"[IR评分] 发现关键词 '{kw}' +3分")

    # 2. 像素格式中出现经典灰度格式
    #   常见：Y8、GREY、Y16、Y10、Y12 等
    gray_fmt_keywords = ['y8', 'grey', 'gray', 'y16', 'y10', 'y12']
    if 'formats:' in text or 'pixel formats:' in text:
        for kw in gray_fmt_keywords:
            if re.search(r'\b' + kw + r'\b', text):
                score += 2
                print(f"[IR评分] 发现灰度格式 '{kw}' +2分")

    # 3. 是否有 mono 描述
    if 'mono' in text:
        score += 2
        print(f"[IR评分] 发现 'mono' 描述 +2分")

    # 4. 如果只有一个 Video Capture 节点并且是灰度，也可加一点分
    if 'video capture' in text and 'capture-mplane' not in text:
        score += 1
        print(f"[IR评分] 单一 Video Capture 节点 +1分")

    print(f"[IR评分] 总分: {score}")
    return score


def find_infrared_camera():
    """
    自动检测并返回红外摄像头设备索引

    返回：
        int or None: 红外摄像头的索引，如果未找到返回 None
    """
    import glob
    import subprocess

    print("\n" + "=" * 60)
    print("[自动检测] 开始扫描红外摄像头...")
    print("=" * 60)

    device_paths = {}
    try:
        video_devices = glob.glob('/dev/video*')
        for device_path in video_devices:
            device_name = device_path.split('/')[-1]
            if device_name[5:].isdigit():
                idx = int(device_name.replace('video', ''))
                device_paths[idx] = device_path
    except Exception as e:
        print(f"[自动检测] 读取设备失败: {e}")
        return None

    print(f"[自动检测] 发现 {len(device_paths)} 个视频设备")

    best_device = None
    best_score = 0

    for idx in sorted(device_paths.keys()):
        device_path = device_paths[idx]
        print(f"\n[自动检测] 检查 {device_path}...")

        try:
            # 先用 OpenCV 测试是否可以打开
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"[自动检测] ✗ {device_path} 无法打开，跳过")
                continue
            cap.release()

            # 获取 v4l2 信息并评分
            result = subprocess.run(
                ['v4l2-ctl', '--device', device_path, '--all'],
                capture_output=True,
                text=True,
                timeout=1
            )

            if result.returncode == 0:
                ir_score = score_ir_device(result.stdout)

                if ir_score > best_score:
                    best_score = ir_score
                    best_device = idx
                    print(f"[自动检测] ✓ 当前最佳红外摄像头: {device_path} (得分: {ir_score})")
            else:
                print(f"[自动检测] ✗ v4l2-ctl 执行失败")

        except Exception as e:
            print(f"[自动检测] ✗ {device_path} 检查异常: {e}")

    print("\n" + "=" * 60)
    if best_device is not None and best_score >= 3:
        print(f"[自动检测] 找到红外摄像头: /dev/video{best_device} (得分: {best_score})")
        print("=" * 60 + "\n")
        return best_device
    else:
        print(f"[自动检测] 未找到红外摄像头 (最高得分: {best_score}，需要 ≥3)")
        print("=" * 60 + "\n")
        return None


class IrisService:
    def __init__(self):
        self.camera_index = 0  # 固定使用索引 0
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.is_iris_detected = False
        self.detection_result = {'detected': False}
        self.lock = threading.Lock()

        # 后台持续识别相关
        self.recognition_cache = []  # 识别结果缓存
        self.is_recognizing = False  # 是否正在后台识别
        self.recognition_lock = threading.Lock()

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

    def start_background_recognition(self, duration=5.0):
        """
        启动后台持续识别任务
        在指定时间内持续识别，缓存所有成功结果

        参数:
            duration: 持续识别的时间（秒）
        """
        if self.is_recognizing:
            print("[后台识别] 已有识别任务在运行")
            return {'success': False, 'error': '已有识别任务在运行'}

        # 清空缓存
        with self.recognition_lock:
            self.recognition_cache = []
            self.is_recognizing = True

        def recognition_worker():
            start_time = time.time()
            attempt_count = 0
            success_count = 0

            print(f"[后台识别] 开始，持续 {duration} 秒")

            while time.time() - start_time < duration:
                if not self.is_running:
                    print("[后台识别] 摄像头已停止")
                    break

                attempt_count += 1

                # 检查是否检测到虹膜
                if self.is_iris_detected:
                    try:
                        # 执行识别
                        result = self._recognize_single_frame()

                        if result['success']:
                            success_count += 1
                            confidence = result.get('confidence', 0)

                            # 缓存结果
                            with self.recognition_lock:
                                self.recognition_cache.append({
                                    'timestamp': time.time(),
                                    'result': result,
                                    'confidence': confidence
                                })

                            print(f"[后台识别] 第 {success_count} 次成功: "
                                  f"{result['user_id']}, 置信度 {confidence:.1f}%")

                            # 如果置信度很高，可以提前结束
                            if confidence >= 95:
                                print(f"[后台识别] 置信度达到 {confidence:.1f}%，提前结束")
                                break

                    except Exception as e:
                        print(f"[后台识别] 识别异常: {e}")

                time.sleep(0.15)  # 每150ms尝试一次，避免过于频繁

            elapsed = time.time() - start_time
            with self.recognition_lock:
                self.is_recognizing = False

            print(f"[后台识别] 结束，耗时 {elapsed:.1f}秒，"
                  f"尝试 {attempt_count} 次，成功 {success_count} 次")

        # 启动后台线程
        threading.Thread(target=recognition_worker, daemon=True).start()
        return {'success': True, 'message': f'后台识别已启动，持续 {duration} 秒'}

    def get_best_recognition_result(self):
        """
        获取缓存中置信度最高的识别结果
        """
        with self.recognition_lock:
            if not self.recognition_cache:
                return {'success': False, 'error': '未检测到虹膜或识别失败'}

            # 按置信度排序，取最高的
            best = max(self.recognition_cache, key=lambda x: x['confidence'])

            print(f"[获取结果] 从 {len(self.recognition_cache)} 个结果中选择最佳: "
                  f"{best['result']['user_id']}, 置信度 {best['confidence']:.1f}%")

            return best['result']

    def _recognize_single_frame(self):
        """
        识别单帧（内部方法，不加锁）
        """
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
                return {
                    'success': True,
                    'user_id': name,
                    'eye': eye,
                    'confidence': round(confidence, 2),
                    'score': best_score
                }

            return {'success': False, 'error': '未找到匹配'}
        except Exception as e:
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
                # 先检查设备能力并计算 IR 评分
                v4l2_output = ''
                ir_score = 0
                try:
                    import subprocess
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', device_path, '--all'],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )

                    if result.returncode == 0:
                        v4l2_output = result.stdout
                        debug_entry['raw_v4l2'] = v4l2_output  # 保留原始输出用于调试

                        # 提取设备能力
                        for line in v4l2_output.split('\n'):
                            if 'Device Caps' in line:
                                debug_entry['device_caps'] = line.strip()
                            if 'Card type' in line:
                                debug_entry['card_type'] = line.split(':', 1)[1].strip()

                        # 计算 IR 评分
                        print(f"\n[摄像头列表] 开始评估 {device_path} 的 IR 得分...")
                        ir_score = score_ir_device(v4l2_output)
                        debug_entry['ir_score'] = ir_score
                        debug_entry['is_nir'] = ir_score >= 3  # 阈值：3分

                        if ir_score >= 3:
                            print(f"[摄像头列表] ✓ {device_path} 识别为红外摄像头 (得分: {ir_score})")
                        else:
                            print(f"[摄像头列表] ○ {device_path} 普通摄像头 (得分: {ir_score})")

                except Exception as e:
                    debug_entry['v4l2_error'] = str(e)
                    print(f"[摄像头列表] v4l2-ctl 执行失败: {e}")

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
                        'name': debug_entry.get('card_type', f'Camera {idx}'),
                        'isNir': debug_entry.get('is_nir', False)  # 是否为红外摄像头
                    }

                    cameras.append(camera_info)
                    cap.release()

                    nir_tag = " [红外]" if camera_info['isNir'] else ""
                    print(f"[摄像头列表] ✓ {device_path} 可用: {camera_info['name']}{nir_tag}")
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


@app.route('/api/detect/start-recognition', methods=['POST'])
def start_recognition():
    """
    启动后台持续识别（用于酒精检测场景）
    前端在用户点击"开始检测"时调用
    """
    data = request.json or {}
    duration = data.get('duration', 5.0)  # 默认持续5秒

    # 确保摄像头运行
    if not iris_service.is_running:
        print("[启动识别] 启动摄像头...")
        iris_service.start_camera()
        time.sleep(0.8)  # 等待摄像头初始化

    # 启动后台识别
    result = iris_service.start_background_recognition(duration)

    return jsonify(result)


@app.route('/api/detect/get-result', methods=['POST'])
def get_recognition_result():
    """
    获取后台识别的最佳结果
    前端在酒精检测完成后调用
    """
    # 等待后台识别完成（最多等待1秒）
    timeout = 1.0
    start_time = time.time()

    while iris_service.is_recognizing:
        if time.time() - start_time > timeout:
            print("[获取结果] 后台识别仍在进行，直接返回当前最佳结果")
            break
        time.sleep(0.1)

    result = iris_service.get_best_recognition_result()

    return jsonify(result)


# ==================== 数据同步 API ====================

@app.route('/api/sync/bidirectional', methods=['POST'])
def sync_bidirectional():
    """
    双向同步接口
    接收两个客户端的地址，自动完成双向数据同步

    请求参数：
    {
        "clientA": "192.168.2.59:8084",
        "clientB": "8.155.146.165:8084"
    }
    """
    try:
        data = request.json or {}
        client_a = data.get('clientA', '')
        client_b = data.get('clientB', '')

        if not client_a or not client_b:
            return jsonify({
                'success': False,
                'error': '缺少客户端地址参数'
            })

        print(f"[双向同步] 开始同步: {client_a} <-> {client_b}")

        # 判断当前服务是哪个客户端
        current_host = request.host  # 例如：192.168.2.59:8084

        if current_host == client_a:
            # 当前是客户端A
            peer_url = f'http://{client_b}'
            print(f"[双向同步] 当前是客户端A，对方是客户端B: {peer_url}")
        elif current_host == client_b:
            # 当前是客户端B
            peer_url = f'http://{client_a}'
            print(f"[双向同步] 当前是客户端B，对方是客户端A: {peer_url}")
        else:
            return jsonify({
                'success': False,
                'error': f'当前服务地址 {current_host} 不在同步列表中'
            })

        # 执行双向同步
        result = perform_bidirectional_sync(peer_url)

        return jsonify(result)

    except Exception as e:
        print(f"[双向同步] 失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/sync/download', methods=['POST'])
def sync_download():
    """
    接收对方客户端推送的虹膜数据
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未找到上传文件'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'})

        print(f"[同步下载] 接收文件: {file.filename}")

        # 保存到临时目录
        zip_path = '/tmp/iris_data_received.zip'
        file.save(zip_path)

        file_size = os.path.getsize(zip_path) / 1024 / 1024
        print(f"[同步下载] 文件大小: {file_size:.2f} MB")

        # 备份当前数据
        backup_dir = f'/tmp/iris_backup_{int(time.time())}'
        os.makedirs(backup_dir, exist_ok=True)

        if os.path.exists('photo'):
            shutil.copytree('photo', os.path.join(backup_dir, 'photo'))
        if os.path.exists('feature'):
            shutil.copytree('feature', os.path.join(backup_dir, 'feature'))

        print(f"[同步下载] 已备份到: {backup_dir}")

        # 解压覆盖
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall('.')

        print("[同步下载] 解压完成")

        # 清理临时文件
        os.remove(zip_path)

        # 清理旧备份（保留最近3个）
        cleanup_old_backups('/tmp', 'iris_backup_', keep=3)

        return jsonify({
            'success': True,
            'message': '虹膜数据同步完成',
            'size': f'{file_size:.2f} MB',
            'backup': backup_dir
        })

    except Exception as e:
        print(f"[同步下载] 失败: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sync/package', methods=['POST'])
def sync_package():
    """
    打包虹膜数据供对方客户端下载
    """
    try:
        print("[打包数据] 开始...")

        timestamp = int(time.time())
        zip_filename = f'iris_data_{timestamp}.zip'
        zip_path = f'/tmp/{zip_filename}'

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 打包 photo 目录
            for root, dirs, files in os.walk('photo'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)

            # 打包 feature 目录
            for root, dirs, files in os.walk('feature'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)

        file_size = os.path.getsize(zip_path)
        print(f"[打包数据] 完成，大小: {file_size / 1024 / 1024:.2f} MB")

        # 返回下载链接
        download_url = f'http://{request.host}/api/sync/download-file/{zip_filename}'

        return jsonify({
            'success': True,
            'url': download_url,
            'size': file_size
        })

    except Exception as e:
        print(f"[打包数据] 失败: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sync/download-file/<filename>', methods=['GET'])
def sync_download_file(filename):
    """
    下载打包好的虹膜数据文件
    """
    try:
        file_path = f'/tmp/{filename}'
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404

        return send_file(
            file_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 辅助函数 ====================

def perform_bidirectional_sync(peer_url):
    """
    执行双向同步逻辑
    1. 推送本地数据到对方
    2. 从对方拉取数据
    """
    try:
        print(f"[双向同步] 目标: {peer_url}")

        # 步骤1：推送本地数据到对方
        print("[双向同步] 步骤1: 推送本地数据到对方...")
        push_result = push_data_to_peer(peer_url)

        if not push_result['success']:
            return {
                'success': False,
                'error': f'推送数据失败: {push_result["error"]}'
            }

        # 步骤2：从对方拉取数据
        print("[双向同步] 步骤2: 从对方拉取数据...")
        pull_result = pull_data_from_peer(peer_url)

        if not pull_result['success']:
            return {
                'success': False,
                'error': f'拉取数据失败: {pull_result["error"]}'
            }

        return {
            'success': True,
            'message': '双向同步完成',
            'details': {
                'push': push_result,
                'pull': pull_result
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def push_data_to_peer(peer_url):
    """推送本地数据到对方客户端"""
    try:
        # 打包本地数据
        zip_path = '/tmp/iris_data_push.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk('photo'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)

            for root, dirs, files in os.walk('feature'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)

        file_size = os.path.getsize(zip_path) / 1024 / 1024
        print(f"[推送数据] 打包完成，大小: {file_size:.2f} MB")

        # 上传到对方
        with open(zip_path, 'rb') as f:
            files = {'file': ('iris_data.zip', f, 'application/zip')}
            response = requests.post(
                f'{peer_url}/api/sync/download',
                files=files,
                timeout=300
            )

        os.remove(zip_path)

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {
                'success': False,
                'error': f'HTTP {response.status_code}'
            }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def pull_data_from_peer(peer_url):
    """从对方客户端拉取数据"""
    try:
        # 请求对方打包数据
        response = requests.post(
            f'{peer_url}/api/sync/package',
            timeout=60
        )

        if response.status_code != 200:
            return {
                'success': False,
                'error': f'对方响应异常: {response.status_code}'
            }

        result = response.json()
        if not result.get('success'):
            return result

        zip_url = result.get('url')
        print(f"[拉取数据] 下载链接: {zip_url}")

        # 下载 ZIP
        zip_response = requests.get(zip_url, stream=True, timeout=300)
        zip_path = '/tmp/iris_data_pull.zip'

        with open(zip_path, 'wb') as f:
            for chunk in zip_response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(zip_path) / 1024 / 1024
        print(f"[拉取数据] 下载完成，大小: {file_size:.2f} MB")

        # 备份并解压
        backup_dir = f'/tmp/iris_backup_{int(time.time())}'
        os.makedirs(backup_dir, exist_ok=True)

        if os.path.exists('photo'):
            shutil.copytree('photo', os.path.join(backup_dir, 'photo'))
        if os.path.exists('feature'):
            shutil.copytree('feature', os.path.join(backup_dir, 'feature'))

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall('.')

        os.remove(zip_path)

        return {
            'success': True,
            'message': '拉取数据成功',
            'size': f'{file_size:.2f} MB'
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def cleanup_old_backups(directory, prefix, keep=3):
    """清理旧备份，保留最近 N 个"""
    try:
        backups = sorted(
            [d for d in os.listdir(directory) if d.startswith(prefix)],
            reverse=True
        )
        for old_backup in backups[keep:]:
            backup_path = os.path.join(directory, old_backup)
            shutil.rmtree(backup_path)
            print(f"[清理备份] 删除: {backup_path}")
    except Exception as e:
        print(f"[清理备份] 失败: {e}")


# ==================== 启动 ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='虹膜识别服务')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8084, help='监听端口 (默认: 8084)')
    parser.add_argument('--camera', type=int, help='手动指定摄像头索引（跳过自动检测）')
    args = parser.parse_args()

    # 自动检测红外摄像头
    camera_index = 0  # 默认值
    if args.camera is not None:
        # 用户手动指定了摄像头索引
        camera_index = args.camera
        print("\n" + "=" * 60)
        print(f"[手动指定] 使用摄像头索引: {camera_index}")
        print("=" * 60 + "\n")
    else:
        # 自动检测红外摄像头
        detected_index = find_infrared_camera()
        if detected_index is not None:
            camera_index = detected_index
            print(f"[自动设置] 检测到红外摄像头已将摄像头索引设置为: {camera_index}")
        else:
            print(f"[默认设置] 未检测到红外摄像头，使用默认索引: {camera_index}")

    # 设置到全局服务实例
    iris_service.camera_index = camera_index

    print("\n" + "=" * 60)
    print(f"虹膜识别服务启动")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"摄像头索引: {camera_index} (未启动)")
    print("")
    print("API 端点:")
    print(f"  - 状态查询: http://{args.host}:{args.port}/api/status")
    print(f"  - 摄像头列表: GET http://{args.host}:{args.port}/api/camera/list")
    print(f"  - 设置摄像头: POST http://{args.host}:{args.port}/api/camera/set-device")
    print(f"  - 启动摄像头: POST http://{args.host}:{args.port}/api/camera/start")
    print(f"  - 停止摄像头: POST http://{args.host}:{args.port}/api/camera/stop")
    print(f"  - 视频流: http://{args.host}:{args.port}/api/video/stream")
    print("")
    print("提示: 摄像头需要手动启动后才能使用")
    print("提示: 可通过 /api/camera/set-device 重新设置摄像头")
    print("=" * 60 + "\n")

    app.run(host=args.host, port=args.port, threaded=True)

