#!/usr/bin/env python3
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import Image  # type: ignore
from robot_msgs.msg import MotionEnd, LineResult, BallPosition 
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer  # type: ignore
import rclpy  # type: ignore
import cv2
import numpy as np
import math
import time
from collections import deque

# ==============================
# 상태 정의 (고정)
# ==============================
WAIT_ARM, SEARCH_CAM1, APPROACH_CAM1, BRIDGE_TO_CAM2, SEARCH_CAM2, ALIGN_CAM2, PICK_READY, FALLBACK_LINE = range(8)

class BasketballTrackerNode(Node):
    """
    단일 노드 (B안)
    - cam1(color+aligned_depth_to_color), cam2(color) 둘 다 구독
    - /motion_end 수신 시에만 '윈도우(15프레임)' 수집/판정/퍼블리시 진행
    - cam1 단계: LineResult(res, angle) 퍼블리시(라인과 동일 규칙)
    - cam2 단계: BallPosition(status, dx, dy, r) 퍼블리시(미세 정렬)
    """

    def __init__(self):
        super().__init__('basketball_tracker')

        # -----------------------------
        # 공용 변수 (이미지/ROI/표시)
        # -----------------------------
        self.image_width  = 640
        self.image_height = 480

        self.roi_x_start = int(self.image_width * 1 // 5)   # 초록 박스 관심 구역
        self.roi_x_end   = int(self.image_width * 4 // 5)
        self.roi_y_start = int(self.image_height * 1 // 12)
        self.roi_y_end   = int(self.image_height * 11 // 12)

        # zandi (기준점)
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 카메라 내참수(표시/참고용)
        self.fx, self.fy = 607.0, 606.0
        self.cx_intr, self.cy_intr = 325.5, 239.4

        # -----------------------------
        # 색/모폴로지/임계 (상수)
        # -----------------------------
        self.lower_hsv = np.array([8, 60, 60],   dtype=np.uint8)   # 주황색 대역(초기)
        self.upper_hsv = np.array([60, 255, 255], dtype=np.uint8)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = 1000.0
        self.max_circle_ratio = 0.3  # |area/circle_area - 1| < 0.3

        # cam1 깊이 제한
        self.depth_scale   = 0.001  # m
        self.depth_max_mm  = 3000.0 # 3m
        self.depth_max_m   = 3.0

        # cam2 근접/정렬 임계
        self.r_near = 25          # px (반지름이 이 이상이면 근접으로 판단)
        self.x_tol  = 12          # px
        self.y_tol  = 12          # px

        # cam1 회전 판단 임계
        self.collecting_frames = 15
        self.vertical   = 15      # deg (직진 vs 회전)
        self.horizontal = 75      # deg (참고용)

        # 브릿지 설정 (cam1 유실 후 cam2 획득까지 전진)
        self.bridge_steps     = 3
        self.bridge_timeout_s = 5.0

        # 디버그 표시 on/off
        self.show_debug = True

        # -----------------------------
        # FSM/윈도우 상태
        # -----------------------------
        self.state = WAIT_ARM
        self.armed = False           # /motion_end 들어와야 다음 윈도우 시작
        self.window_id = 0
        self.frames_left = 0
        self.win_start_time = None
        self.frame_idx = 0

        self.bridge_left = 0
        self.bridge_start_t = 0.0

        # 버퍼 (현재 상태의 카메라만 누적)
        self.cam1_buf = deque(maxlen=self.collecting_frames)  # [{'found':bool,'cx':..,'cy':..,'z':..,'r':..,'a':..}]
        self.cam2_buf = deque(maxlen=self.collecting_frames)

        # 딜레이/FPS 표시용
        self.cam1_frame_count = 0
        self.cam2_frame_count = 0
        self.total_time_cam1 = 0.0
        self.total_time_cam2 = 0.0
        self.last_report_time_cam1 = time.time()
        self.last_report_time_cam2 = time.time()
        self.last_avg_text_cam1 = 'AVG: --- ms | FPS: --'
        self.last_avg_text_cam2 = 'AVG: --- ms | FPS: --'

        # 화면용 최근 값
        self.last_position_text_cam1 = 'Dist: -- m | Pos: --, --'
        self.last_position_text_cam2 = 'Pos: --, --'
        self.ball_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)

        # -----------------------------
        # 구독/퍼블리시
        # -----------------------------
        self.bridge = CvBridge()

        # cam1 동기화(정면): /cam1/color/image_raw + /cam1/aligned_depth_to_color/image_raw
        sub_c1_color = Subscriber(self, Image, '/cam1/color/image_raw')
        sub_c1_depth = Subscriber(self, Image, '/cam1/aligned_depth_to_color/image_raw')
        self.cam1_sync = ApproximateTimeSynchronizer([sub_c1_color, sub_c1_depth], queue_size=5, slop=0.1)
        self.cam1_sync.registerCallback(self.cam1_image_callback)

        # cam2(바닥): /cam2/color/image_raw
        self.sub_cam2 = self.create_subscription(Image, '/cam2/color/image_raw', self.cam2_image_callback, 10)

        # 모션 동기: /motion_end
        self.sub_me = self.create_subscription(MotionEnd, '/motion_end', self.motion_callback, 10)

        # 출력
        self.pub_cam1 = self.create_publisher(LineResult,  '/ball_result_cam1', 10)  # res, angle
        self.pub_cam2 = self.create_publisher(BallPosition, '/ball_adjust_cam2', 10)   # status, dx, dy, r

        # 디버그 창
        if self.show_debug:
            cv2.namedWindow('cam1/Mask')
            cv2.namedWindow('cam1/Final')
            cv2.namedWindow('cam1/Frame')
            cv2.namedWindow('cam2/Mask')
            cv2.namedWindow('cam2/Final')
            cv2.namedWindow('cam2/Frame')

        self.get_logger().info("[Tracker] Ready. WAIT_ARM")

    # ==============================
    # /motion_end → 윈도우 시작
    # ==============================
    def motion_callback(self, msg: MotionEnd):
        if not bool(msg.motion_end_detect):
            return
        self.armed = True
        # 상태 진입 초기화
        if self.state == WAIT_ARM:
            self.state = SEARCH_CAM1
        # 브릿지 타임아웃
        if self.state == BRIDGE_TO_CAM2 and (time.time() - self.bridge_start_t) > self.bridge_timeout_s:
            self.state = FALLBACK_LINE
            self.get_logger().warn("[BRIDGE] timeout → FALLBACK_LINE")
        self._start_window()

    def _start_window(self):
        if not self.armed:
            return
        self.window_id += 1
        self.frames_left = self.collecting_frames
        self.cam1_buf.clear()
        self.cam2_buf.clear()
        self.frame_idx = 0
        self.win_start_time = time.time()
        self.get_logger().info(f"[Window] #{self.window_id} start | state={self._sname(self.state)}")

    # ==============================
    # cam1 콜백 (정면, color+depth)
    # ==============================
    def cam1_image_callback(self, color_msg: Image, depth_msg: Image):
        t0 = time.time()
        # 윈도우 수집 중이고 cam1 상태일 때만 처리
        if self.frames_left <= 0 or self.state not in (SEARCH_CAM1, APPROACH_CAM1):
            return

        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        roi_depth = depth[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        raw_mask[roi_depth >= self.depth_max_mm] = 0  # 3m 초과 제거

        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None; best_ratio = 1.0
        x_best = y_best = None
        radius = 0.0; area = 0.0

        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > self.min_area:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                circ_area = math.pi * r * r
                ratio = abs((a / (circ_area + 1e-6)) - 1.0)
                if ratio < self.max_circle_ratio and ratio < best_ratio:
                    best_ratio = ratio
                    best_cnt = cnt
                    x_best, y_best = int(x), int(y)
                    radius = float(r); area = float(a)

        found = False; cx = cy = 0.0; z_m = -1.0
        if best_cnt is not None:
            cx_ball = x_best + self.roi_x_start
            cy_ball = y_best + self.roi_y_start

            # 깊이 평균 (3x3)
            x1 = max(x_best - 1, 0)
            x2 = min(x_best + 2, self.roi_x_end - self.roi_x_start)
            y1 = max(y_best - 1, 0)
            y2 = min(y_best + 2, self.roi_y_end - self.roi_y_start)
            roi_patch = roi_depth[y1:y2, x1:x2]
            z_m = float(np.mean(roi_patch)) * self.depth_scale if roi_patch.size > 0 else -1.0

            if 0.0 < z_m <= self.depth_max_m:
                found = True
                cx = float(cx_ball); cy = float(cy_ball)

        # 버퍼 누적
        self.cam1_buf.append({'found':found,'cx':cx,'cy':cy,'z':z_m,'r':radius,'a':area})
        self.frames_left -= 1
        self.frame_idx += 1

        # 디버그
        if self.show_debug:
            final_mask = np.zeros_like(mask)
            if best_cnt is not None:
                cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)

            elapsed = time.time() - t0
            self.cam1_frame_count += 1
            self.total_time_cam1 += elapsed
            now = time.time()
            if now - self.last_report_time_cam1 >= 1.0:
                avg_time = self.total_time_cam1 / max(1, self.cam1_frame_count)
                fps = self.cam1_frame_count / max(1e-6, (now - self.last_report_time_cam1))
                self.last_avg_text_cam1 = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
                self.cam1_frame_count = 0; self.total_time_cam1 = 0.0; self.last_report_time_cam1 = now

            vis = frame.copy()
            cv2.rectangle(vis, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), (0,255,0), 1)
            cv2.circle(vis, (self.zandi_x, self.zandi_y), 3, (255,255,255), -1)
            if found:
                cv2.circle(vis, (int(cx), int(cy)), int(radius), (255,0,0), 2)
                cv2.circle(vis, (int(cx), int(cy)), 4, (0,0,255), -1)
                self.last_position_text_cam1 = f'Dist: {z_m:.2f}m | Pos: {int(cx - self.zandi_x)}, {int(-(cy - self.zandi_y))}'
            cv2.putText(vis, self.last_avg_text_cam1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(vis, f"Step: {self.frame_idx}/{self.collecting_frames}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(vis, self.last_position_text_cam1, (self.roi_x_start - 175, self.roi_y_end + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,30), 2)
            cv2.imshow('cam1/Mask', mask)
            cv2.imshow('cam1/Final', final_mask)
            cv2.imshow('cam1/Frame', vis)
            cv2.waitKey(1)

        # 윈도우 종료
        if self.frames_left == 0:
            self._finish_window()

    # ==============================
    # cam2 콜백 (바닥, color only)
    # ==============================
    def cam2_image_callback(self, msg: Image):
        t0 = time.time()
        # 윈도우 수집 중이고 cam2 관련 상태일 때만 처리
        if self.frames_left <= 0 or self.state not in (BRIDGE_TO_CAM2, SEARCH_CAM2, ALIGN_CAM2):
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None; best_ratio = 1.0
        x_best = y_best = None
        radius = 0.0; area = 0.0

        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > self.min_area:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                circ_area = math.pi * r * r
                ratio = abs((a / (circ_area + 1e-6)) - 1.0)
                if ratio < self.max_circle_ratio and ratio < best_ratio:
                    best_ratio = ratio
                    best_cnt = cnt
                    x_best, y_best = int(x), int(y)
                    radius = float(r); area = float(a)

        found = False; cx = cy = 0.0
        if best_cnt is not None:
            cx_ball = x_best + self.roi_x_start
            cy_ball = y_best + self.roi_y_start
            found = True
            cx = float(cx_ball); cy = float(cy_ball)

        self.cam2_buf.append({'found':found,'cx':cx,'cy':cy,'z':-1.0,'r':radius,'a':area})
        self.frames_left -= 1
        self.frame_idx += 1

        # 디버그
        if self.show_debug:
            final_mask = np.zeros_like(mask)
            if best_cnt is not None:
                cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)

            elapsed = time.time() - t0
            self.cam2_frame_count += 1
            self.total_time_cam2 += elapsed
            now = time.time()
            if now - self.last_report_time_cam2 >= 1.0:
                avg_time = self.total_time_cam2 / max(1, self.cam2_frame_count)
                fps = self.cam2_frame_count / max(1e-6, (now - self.last_report_time_cam2))
                self.last_avg_text_cam2 = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
                self.cam2_frame_count = 0; self.total_time_cam2 = 0.0; self.last_report_time_cam2 = now

            vis = frame.copy()
            cv2.rectangle(vis, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), (0,255,0), 1)
            cv2.circle(vis, (self.zandi_x, self.zandi_y), 3, (255,255,255), -1)
            if found:
                cv2.circle(vis, (int(cx), int(cy)), int(radius), (255,0,0), 2)
                cv2.circle(vis, (int(cx), int(cy)), 4, (0,0,255), -1)
                self.last_position_text_cam2 = f'Pos: {int(cx - self.zandi_x)}, {int(-(cy - self.zandi_y))}'
            cv2.putText(vis, self.last_avg_text_cam2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(vis, f"Step: {self.frame_idx}/{self.collecting_frames}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(vis, self.last_position_text_cam2, (self.roi_x_start - 175, self.roi_y_end + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,30), 2)
            cv2.imshow('cam2/Mask', mask)
            cv2.imshow('cam2/Final', final_mask)
            cv2.imshow('cam2/Frame', vis)
            cv2.waitKey(1)

        if self.frames_left == 0:
            self._finish_window()

    # ==============================
    # 윈도우 종료 → 상태별 판정
    # ==============================
    def _finish_window(self):
        dur_ms = (time.time() - self.win_start_time) * 1000.0 if self.win_start_time else 0.0

        if self.state in (SEARCH_CAM1, APPROACH_CAM1):
            # cam1 집계
            vals = [o for o in self.cam1_buf if o['found']]
            if len(vals) > 0:
                cx = float(np.mean([v['cx'] for v in vals]))
                cy = float(np.mean([v['cy'] for v in vals]))
                res, ang = self._decide_cam1(cx, cy)
                out = LineResult()
                out.res = res
                out.angle = abs(int(round(ang)))
                self.pub_cam1.publish(out)
                self.get_logger().info(f"[CAM1] res={res}, angle={out.angle} | {dur_ms:.1f}ms")
                self.state = APPROACH_CAM1
            else:
                # cam1 유실 → 브릿지 모드
                self.state = BRIDGE_TO_CAM2
                self.bridge_left = self.bridge_steps
                self.bridge_start_t = time.time()
                self.get_logger().warn(f"[CAM1] MISS → BRIDGE_TO_CAM2 (steps={self.bridge_left})")

        elif self.state == BRIDGE_TO_CAM2:
            # cam2 획득 체크 + 전진 명령 1회
            got, dx, dy, r = self._aggregate_cam2()
            if got and r >= self.r_near:
                self.state = SEARCH_CAM2
                self.get_logger().info(f"[BRIDGE] cam2 DETECT (r={r:.1f}) → SEARCH_CAM2")
            else:
                out = LineResult()  # 라인과 동일: 직진 명령 (한 스텝)
                out.res = 1
                out.angle = 0
                self.pub_cam1.publish(out)
                self.bridge_left -= 1
                self.get_logger().info(f"[BRIDGE] Forward step | left={self.bridge_left} | {dur_ms:.1f}ms")
                if self.bridge_left <= 0 or (time.time() - self.bridge_start_t) > self.bridge_timeout_s:
                    self.state = FALLBACK_LINE
                    self.get_logger().warn("[BRIDGE] done/timeout → FALLBACK_LINE")

        elif self.state == SEARCH_CAM2:
            got, dx, dy, r = self._aggregate_cam2()
            if got and r >= self.r_near:
                self.state = ALIGN_CAM2
                self.get_logger().info(f"[CAM2] DETECT (r={r:.1f}) → ALIGN_CAM2")
            else:
                self.state = FALLBACK_LINE
                self.get_logger().warn("[CAM2] MISS → FALLBACK_LINE")

        elif self.state == ALIGN_CAM2:
            got, dx, dy, r = self._aggregate_cam2()
            adj = BallPosition()
            if got:
                adj.dx = int(round(dx))
                adj.dy = int(round(dy))
                adj.r  = int(round(r))
                if abs(dx) <= self.x_tol and abs(dy) <= self.y_tol:
                    adj.status = 1  # ALIGNED
                    self.state = PICK_READY
                    self.get_logger().info(f"[ALIGN] ALIGNED | dx={dx:.1f}, dy={dy:.1f} | {dur_ms:.1f}ms")
                else:
                    adj.status = 0  # RUN
                    self.get_logger().info(f"[ALIGN] RUN | dx={dx:.1f}, dy={dy:.1f}, r={r:.1f} | {dur_ms:.1f}ms")
            else:
                adj.status = 2  # MISS
                adj.dx = adj.dy = adj.r = 0
                self.get_logger().warn(f"[ALIGN] MISS | {dur_ms:.1f}ms")
            self.pub_cam2.publish(adj)

        elif self.state == PICK_READY:
            self.get_logger().info("[PICK_READY] 집기 준비 완료")
            self.state = WAIT_ARM

        elif self.state == FALLBACK_LINE:
            self.get_logger().info("[FALLBACK_LINE] 라인트레이싱 복귀")
            self.state = WAIT_ARM

        # 윈도우 리셋 (다음 /motion_end까지 대기)
        self.armed = False
        self.frames_left = 0
        self.cam1_buf.clear()
        self.cam2_buf.clear()
        self.win_start_time = None
        self.frame_idx = 0

    # cam2 집계 유틸
    def _aggregate_cam2(self):
        vals = [o for o in self.cam2_buf if o['found']]
        if len(vals) == 0:
            return False, 0.0, 0.0, 0.0
        cx = float(np.mean([v['cx'] for v in vals]))
        cy = float(np.mean([v['cy'] for v in vals]))
        r  = float(np.mean([v['r']  for v in vals]))
        dx = cx - self.zandi_x
        dy = cy - self.zandi_y
        return True, dx, dy, r

    # cam1 회전 결정 (라인과 동일 규칙)
    def _decide_cam1(self, cx, cy):
        dx = cx - self.zandi_x
        dy = cy - self.zandi_y
        angle = math.degrees(math.atan2(dx, -dy))  # (-90, 90] 범위로 정규화
        if angle > 90: angle -= 180
        if angle <= -90: angle += 180
        if abs(angle) <= self.vertical:
            return 1, angle  # Straight
        elif angle > self.vertical:
            return 3, angle  # Spin Right
        else:
            return 2, angle  # Spin Left

    @staticmethod
    def _sname(s):
        return ["WAIT_ARM","SEARCH_CAM1","APPROACH_CAM1","BRIDGE_TO_CAM2",
                "SEARCH_CAM2","ALIGN_CAM2","PICK_READY","FALLBACK_LINE"][s]

def main():
    rclpy.init()
    node = BasketballTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()