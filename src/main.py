"""main.py — 批量视频处理主入口

功能：遍历 videos/ 目录，对每场比赛视频运行完整追踪流水线，支持断点续跑
升级：集成了防死锁、多格式兼容的空间/姿态拉力状态检测器 (Spatial Rally Detector)
"""
import cv2
import numpy as np
import json
import os
import time
import threading
import queue
import torch
from ultralytics import YOLO

import config_legacy as config
from court_detector import CourtDetector
from pose_tracker import PoseTracker


class SpatialRallyDetector:
    def __init__(self, fps=30, buffer_seconds=1.5, movement_threshold=3.0):
        """
        根据球员的空间分布与移动状态，精准检测每一帧是处于击球对攻阶段（PLAYING）还是死球阶段（NOT PLAYING）
        """
        self.fps = fps
        self.buffer_frames = int(fps * buffer_seconds)
        self.movement_threshold = movement_threshold
        
        self.player_history = []  # 记录前几帧的球员中心点位置
        self.frames_since_active = self.buffer_frames
        self.is_playing = False

    def _extract_box(self, player_data):
        """
        增强型防御性解包：确保无论 PoseTracker 返回字典、对象还是嵌套列表，均能稳定提取出 4 元素边界框
        """
        if not player_data:
            return None
        try:
            # 情况 1：标准字典格式
            if isinstance(player_data, dict):
                box = player_data.get("box")
                if box is not None:
                    if hasattr(box, "tolist"): box = box.tolist()  # 兼容 numpy 数组
                    if isinstance(box, list) and len(box) >= 4:
                        return box[:4]
            # 情况 2：带有 .box 属性的对象
            elif hasattr(player_data, "box"):
                box = player_data.box
                if hasattr(box, "tolist"): box = box.tolist()
                if hasattr(box, "__iter__") and len(box) >= 4:
                    return list(box)[:4]
        except Exception:
            pass
        return None

    def update(self, far_player_data, near_player_data):
        """
        核心逻辑：根据当前帧两边球员的位置与运动幅度更新状态
        """
        is_frame_active = False

        far_box = self._extract_box(far_player_data)
        near_box = self._extract_box(near_player_data)

        # 1. 空间校验：场上必须同时识别到远端和近端球员
        if far_box is not None and near_box is not None:
            try:
                # 计算两个球员在全球坐标系下的中心点
                f_cx = (far_box[0] + far_box[2]) / 2
                f_cy = (far_box[1] + far_box[3]) / 2
                n_cx = (near_box[0] + near_box[2]) / 2
                n_cy = (near_box[1] + near_box[3]) / 2

                current_centers = np.array([[f_cx, f_cy], [n_cx, n_cy]])
                self.player_history.append(current_centers)
                
                if len(self.player_history) > 4:
                    self.player_history.pop(0)

                # 2. 运动校验：计算连续帧之间球员的移动速度 (Velocity)
                if len(self.player_history) >= 2:
                    # 计算前后两帧远近端球员的位移
                    disp_far = np.linalg.norm(self.player_history[-1][0] - self.player_history[-2][0])
                    disp_near = np.linalg.norm(self.player_history[-1][1] - self.player_history[-2][1])
                    max_displacement = max(disp_far, disp_near)

                    # 如果球员处于高频折返、大范围奔跑，则判定此帧为活跃对攻帧
                    if max_displacement > self.movement_threshold:
                        is_frame_active = True
                        
                # 3. 纵向相对位置校验（防并排或过于靠近的死球状态误判，阈值缩减至 50 更弹性）
                if abs(f_cy - n_cy) < 50:  
                    is_frame_active = False
            except Exception:
                is_frame_active = False
        else:
            # 球员数据丢失（如走出画面），默认不活跃
            is_frame_active = False

        # 4. 时间平滑缓冲控制状态切换
        if is_frame_active:
            self.is_playing = True
            self.frames_since_active = 0
        else:
            self.frames_since_active += 1

        if self.frames_since_active >= self.buffer_frames:
            self.is_playing = False
            self.player_history.clear()

        return self.is_playing


class BatchTennisPipeline:
    def __init__(self):
        self.input_dir = config.VIDEO_PATH
        self.output_base_dir = config.OUTPUT_DIR

        # 获取目录下所有 mp4 文件并排序，确保处理顺序一致
        self.video_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith('.mp4')])
        if not self.video_files:
            raise FileNotFoundError(f"在 {self.input_dir} 中未找到任何 mp4 文件！")

        self.court_detector = CourtDetector(scale=config.SCOUT_SCALE)

        torch.backends.cudnn.benchmark = True

        # 全局进度状态
        self.current_video_idx = 0
        self.current_scout_frame = 0
        self.current_task_count = 0
        self.pending_queue_data = []  # 暂存断点时的队列数据

        self._load_checkpoint()

    def _load_checkpoint(self):
        """ 加载本地存档，恢复到特定的视频和特定的帧位 """
        if os.path.exists(config.CHECKPOINT_FILE):
            try:
                with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.current_video_idx = data.get("video_idx", 0)
                self.current_scout_frame = data.get("scout_frame", 0)
                self.current_task_count = data.get("gpu_task_count", 0)
                self.pending_queue_data = data.get("pending_queue", [])

                if self.current_video_idx >= len(self.video_files):
                    print("[*] 存档显示的视频已全部处理完毕，将从头开始。")
                    self.current_video_idx = 0
                    self.current_scout_frame = 0
                    self.current_task_count = 0
                    self.pending_queue_data = []
                else:
                    resume_video = self.video_files[self.current_video_idx]
                    print(f"[*] 读取存档成功。准备继续处理: {resume_video}")
                    print(f"[*] 进度 -> CPU 帧位: {self.current_scout_frame}, GPU 已完成: {self.current_task_count}")
            except Exception:
                print("[!] 存档文件损坏，将从头开始运行。")

    def _save_checkpoint(self):
        """ 挂起时保存跨文件全局状态 """
        pending = list(self.task_queue.queue)
        state = {
            "video_idx": self.current_video_idx,
            "scout_frame": self.current_scout_frame,
            "gpu_task_count": self.current_task_count,
            "pending_queue": pending
        }
        with open(config.CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        print(f"[*] 进度已导出至 {config.CHECKPOINT_FILE}")

    def producer_scout_thread(self, video_path, total_frames, fps, width, height):
        print(f"[CPU] 巡视器启动 -> {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_scout_frame)

        is_active = False
        rally_start_frame = 0
        hits, misses = 0, 0
        current_far_rois, current_near_rois = [], []
        frame_idx = self.current_scout_frame

        HIT_BUF, MISS_BUF = 3, 6

        while cap.isOpened():
            if self.stop_event.is_set():
                self.current_scout_frame = frame_idx
                break

            ret, frame = cap.read()
            if not ret: break

            if frame_idx % config.SCOUT_SKIP_FRAMES == 0:
                far_roi, near_roi = self.court_detector.get_rois(frame, width, height)

                if far_roi is not None:
                    hits += 1
                    misses = 0
                    current_far_rois.append(far_roi)
                    current_near_rois.append(near_roi)

                    if hits > HIT_BUF and not is_active:
                        is_active = True
                        rally_start_frame = max(0, frame_idx - (HIT_BUF * config.SCOUT_SKIP_FRAMES))
                else:
                    misses += 1
                    hits = 0
                    if misses > MISS_BUF and is_active:
                        is_active = False
                        true_end = frame_idx - (MISS_BUF * config.SCOUT_SKIP_FRAMES)
                        duration = (true_end - rally_start_frame) / fps

                        if duration >= config.MIN_RALLY_DURATION:
                            task = {
                                'start': rally_start_frame,
                                'end': true_end,
                                'duration': duration,
                                'far_roi': np.median(current_far_rois, axis=0).astype(int).tolist(),
                                'near_roi': np.median(current_near_rois, axis=0).astype(int).tolist()
                            }
                            self.task_queue.put(task)
                            print(
                                f"[CPU] 回合入队 | 时长: {duration:.1f}s | 进度: {(frame_idx / total_frames) * 100:.1f}%")

                        current_far_rois.clear()
                        current_near_rois.clear()

            frame_idx += 1

        self.current_scout_frame = frame_idx
        cap.release()
        self.scout_finished.set()

    def consumer_yolo_thread(self, video_path, video_output_dir, fps, width, height):
        print("[GPU] 提取机启动")
        model = YOLO(config.MODEL_PATH)
        
        # 智能设备硬件适配加速优化
        if torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f"[GPU] YOLO11 推理硬件自动绑定至: {device.upper()}")
        model.to(device)
        
        tracker = PoseTracker(model)

        # 初始化新增的空间拉力检测模块
        rally_detector = SpatialRallyDetector(fps=fps, buffer_seconds=1.5, movement_threshold=3.0)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        task_count = self.current_task_count

        while True:
            if self.stop_event.is_set():
                self.current_task_count = task_count
                print("[GPU] 收到挂起指令，完成当前片段后安全退出")
                break

            try:
                rally = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                if self.scout_finished.is_set():
                    break
                continue

            task_count += 1
            duration = rally['duration']

            clip_name = f"rally_{task_count:03d}_{duration:.1f}s"
            clip_dir = os.path.join(video_output_dir, clip_name)
            os.makedirs(clip_dir, exist_ok=True)

            raw_path = os.path.join(clip_dir, "raw_clip.mp4")
            ann_path = os.path.join(clip_dir, "annotated_clip.mp4")
            json_path = os.path.join(clip_dir, "pose_data.json")

            out_raw = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
            out_ann = cv2.VideoWriter(ann_path, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, rally['start'])
            curr_frame = rally['start']
            json_data = []

            fx1, fy1, fx2, fy2 = rally['far_roi']
            nx1, ny1, nx2, ny2 = rally['near_roi']

            h_far = {'box': None, 'kpts': None, 'miss': 0}
            h_near = {'box': None, 'kpts': None, 'miss': 0}

            print(f"[GPU] 正在标注: {clip_name}")

            while curr_frame <= rally['end']:
                ret, frame = cap.read()
                if not ret: break

                out_raw.write(frame)
                ann_frame = frame.copy()
                f_data = {"frame": curr_frame, "far_player": None, "near_player": None}

                # 提取追踪当前帧的球员姿态及边框数据
                f_data["far_player"] = tracker.process_and_smooth(
                    frame[fy1:fy2, fx1:fx2], fx1, fy1, True, h_far, ann_frame)

                f_data["near_player"] = tracker.process_and_smooth(
                    frame[ny1:ny2, nx1:nx2], nx1, ny1, False, h_near, ann_frame)

                # 新增核心：调用解包加固的安全检测器
                in_play = rally_detector.update(f_data["far_player"], f_data["near_player"])
                f_data["in_play"] = in_play

                # 在可视化的视频流上渲染状态标志栏
                status_text = "STATE: PLAYING" if in_play else "STATE: NOT PLAYING"
                status_color = (0, 255, 0) if in_play else (0, 0, 255)  # 绿(打球) vs 红(未打球)
                
                # 在左上角画一个状态面板底框并渲染文字
                cv2.rectangle(ann_frame, (30, 30), (450, 100), (15, 15, 15), -1)
                cv2.putText(ann_frame, status_text, (50, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3, cv2.LINE_AA)

                out_ann.write(ann_frame)
                json_data.append(f_data)
                curr_frame += 1

            out_raw.release()
            out_ann.release()
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.current_task_count = task_count
            self.task_queue.task_done()
            print(f"[GPU] 片段完成: {clip_name}")

        cap.release()

    def process_single_video(self, video_path):
        """ 处理单个视频的完整生命周期 """
        temp_cap = cv2.VideoCapture(video_path)
        fps = temp_cap.get(cv2.CAP_PROP_FPS)
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap.release()

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # 重置当前视频的队列与事件
        self.task_queue = queue.Queue()
        for item in self.pending_queue_data:
            self.task_queue.put(item)
        self.pending_queue_data = []  # 装载后清空缓冲

        self.scout_finished = threading.Event()
        self.stop_event = threading.Event()

        scout_t = threading.Thread(target=self.producer_scout_thread,
                                   args=(video_path, total_frames, fps, width, height))
        yolo_t = threading.Thread(target=self.consumer_yolo_thread,
                                  args=(video_path, video_output_dir, fps, width, height))

        scout_t.start()
        yolo_t.start()

        # 监控 control.txt
        while scout_t.is_alive() or yolo_t.is_alive():
            if os.path.exists(config.CONTROL_FILE):
                try:
                    with open(config.CONTROL_FILE, "r", encoding="utf-8") as f:
                        cmd = f.read().strip().lower()
                    if cmd == "save":
                        print("\n[*] 接收到保存指令，正在挂起工作线程...")
                        self.stop_event.set()
                        with open(config.CONTROL_FILE, "w", encoding="utf-8") as f:
                            f.write("saved")
                        break
                except Exception:
                    pass
            time.sleep(2)

        scout_t.join()
        yolo_t.join()

        return self.stop_event.is_set()

    def run(self):
        s_time = time.time()

        for idx in range(self.current_video_idx, len(self.video_files)):
            self.current_video_idx = idx
            video_file = self.video_files[idx]
            video_path = os.path.join(self.input_dir, video_file)

            print(f"\n{'=' * 50}")
            print(f"[*] 开始处理队列 ({idx + 1}/{len(self.video_files)}): {video_file}")
            print(f"{'=' * 50}")

            is_stopped = self.process_single_video(video_path)

            if is_stopped:
                self._save_checkpoint()
                print(f"[*] 断点已成功保存。随时可以安全关闭程序。")
                return
            else:
                print(f"[*] 视频 {video_file} 处理完毕。")
                self.current_scout_frame = 0
                self.current_task_count = 0

        print(f"\n[!!!] 文件夹内所有视频批量处理完成 [!!!]")
        print(f"总耗时: {(time.time() - s_time) / 60:.2f} 分钟。")
        if os.path.exists(config.CHECKPOINT_FILE):
            os.remove(config.CHECKPOINT_FILE)


if __name__ == '__main__':
    pipeline = BatchTennisPipeline()
    pipeline.run()