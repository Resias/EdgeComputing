import threading
import requests
import struct
import time
import queue


import cv2
import numpy as np

import tkinter as tk
from PIL import Image, ImageTk

# ======================================
# 설정 부분
# ======================================

# RTSP 스트림 URL (반드시 정확히 입력)
RTSP_URL = "rtsp://root:root@169.254.248.57:554/cam0_0"

# HTTP Snapshot용 IP (JPEG 메타데이터 파싱에만 사용)
CAM_IP = "169.254.248.57"

# 클래스 ID → 라벨 매핑
def get_label_name(classid: int):
    if classid == 15:
        return "Person"
    else:
        return "No Person"

# ======================================
# AI 메타데이터 파싱 함수 (JPEG 바이트에서)
# ======================================

def parse_ai_objects(ai_data: bytes):
    """
    ai_data: JPEG 내부 AI 메타데이터(232바이트)만 잘라낸 바이트 배열
    12바이트 단위로 (classid, confidence, x, y, w, h) 튜플을 추출하여 리스트로 반환
    """
    objs = []
    for i in range(0, len(ai_data), 12):
        chunk = ai_data[i : i + 12]
        if len(chunk) < 12:
            break
        classid, confidence, x, y, w, h = struct.unpack('<HHHHHH', chunk)
        if confidence == 0:
            continue
        objs.append({
            'classid': classid,
            'confidence': confidence,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        })
    return objs

def get_ai_info_from_bytes(jpeg_bytes: bytes):
    """
    JPEG 바이트 전체(인코딩된 프레임)에서 AI 메타데이터(232바이트 영역)만 추출해 parse_ai_objects() 호출.
    - JPEG 헤더 offset 48~51에서 파일 크기 정보를 읽어 실제 길이와 비교
    - offset 86부터 232바이트 구간을 parse_ai_objects()에 전달
    """
    try:
        data = jpeg_bytes
        if len(data) < 318:
            return []
        # JPEG 헤더 offset(48~51)에 저장된 전체 파일 크기
        n_file_size = (data[48] << 24) | (data[49] << 16) | (data[50] << 8) | data[51]
        if n_file_size != len(data):
            # 헤더가 기대하는 크기와 실제 데이터 크기가 다르면 메타데이터 없다고 간주
            return []
        # AI 데이터가 offset 86부터 232바이트
        ai_data = data[86 : 86 + 232]
        return parse_ai_objects(ai_data)
    except Exception as e:
        print(f"[EXCEPTION] Get AI Info 에러: {e}")
        return []

# ======================================
# HTTP로 JPEG Snapshot 가져오기
# ======================================

def fetch_jpeg_bytes(ip: str, user: str = "root", password: str = "root", timeout_sec: int = 5):
    """
    HTTP Snapshot URL에서 JPEG 바이트 한 장을 메모리로 가져옴.
    return: JPEG 전체 바이트(b'...') 혹은 None(실패)
    """
    try:
        url = f"http://{ip}/cgi-bin/fwcamimg.cgi?FwModId=0&PortId=3&FwCgiVer=0x0001"
        response = requests.get(url, auth=(user, password), timeout=timeout_sec, stream=True)
        if response.status_code == 200:
            return response.content
        else:
            print(f"[ERROR] HTTP Snapshot Response {response.status_code}")
            return None
    except Exception as e:
        print(f"[EXCEPTION] Failed to fetch JPEG: {e}")
        return None




def get_human() -> int:
    """
    HTTP 스냅샷에서 AI 메타데이터를 파싱하고,
    classid == 15 (Person)인 객체 개수를 반환
    """
    jpeg = fetch_jpeg_bytes(CAM_IP)
    if not jpeg:
        return 0
    objs = get_ai_info_from_bytes(jpeg)
    count = sum(1 for obj in objs if obj['classid'] == 15)
    return count



# ======================================
# RTSP 스트리밍 전용 스레드
# ======================================


class StreamThread(threading.Thread):
    def __init__(self, frame_queue: queue.Queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = False
        self.cap = None

    def run(self):
        # 1) RTSP 스트림 열기
        self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"[ERROR] RTSP 스트림을 열 수 없습니다: {RTSP_URL}")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # 스트림 읽기에 실패하면 잠시 대기 후 재시도
                cv2.waitKey(10)
                continue

            # 2) 1280×640으로 리사이즈 (GUI 상단 영역 크기)
            resized = cv2.resize(frame, (640, 720), interpolation=cv2.INTER_LANCZOS4)

            # 3) 프레임 큐에 최신 프레임만 남기도록
            if not self.frame_queue.empty():
                try:
                    # 이전에 남은 프레임은 더 이상 사용하지 않으므로 폐기
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(resized)

            # 5) 아주 짧게 대기 (너무 짧으면 CPU 과부하, 길면 화면 끊김)
            time.sleep(0.01)

        # 루프 중단 시 RTSP 스트림 해제
        # 종료 시 자원 해제
        if self.cap:
            self.cap.release()


# ======================================
# 메타데이터 파싱 전용 스레드
# ======================================

# class MetadataThread(threading.Thread):
#     def __init__(self, update_callback):
#         super().__init__()
#         self.update_callback = update_callback  # 경고문 문자열을 GUI에 표시할 콜백
#         self.running = False

#     def run(self):
#         self.running = True
#         while self.running:
#             # 1) HTTP Snapshot으로 JPEG 바이트 가져오기
#             jpeg_bytes = fetch_jpeg_bytes(CAM_IP)
#             warning_text = ""
#             if jpeg_bytes:
#                 # 2) JPEG 바이트에서 AI 메타데이터 파싱
#                 objs = get_ai_info_from_bytes(jpeg_bytes)

#                 # 3) 파싱된 객체 리스트로부터 라벨 모음 생성
#                 labels = { get_label_name(obj['classid']) for obj in objs }

#                 # 4) 각 라벨에 대응하는 한글 경고문 합치기
#                 if labels:
#                     messages = [WARNING_TEXT[label] for label in labels if label in WARNING_TEXT]
#                     warning_text = "\n".join(messages)

#             # 5) GUI에 경고문 전달 (빈 문자열일 경우 화면에서 사라짐)
#             self.update_callback(warning_text)

#             # 6) 1초 간격으로 반복
#             time.sleep(1.0)

# ======================================
# Tkinter GUI 설정
# ======================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Camera GUI (1280×720)")
        # 크기 1280×720 고정
        self.root.geometry("1280x720")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # (1) 비디오 표시용 Label: 상단 1280×640
        self.video_label = tk.Label(self.root)
        self.video_label.place(x=0, y=0, width=640, height=640)


        # 스레드 및 이미지 저장 변수
        self.frame_queue = queue.Queue(maxsize=1)
        self.stream_thread = None
        self.photo_image = None  # 최신 프레임 저장용
        
        # (2) 스트림 업데이트 예약
        self._schedule_video_update()
        
        # (3) 스트리밍 시작 버튼 (테스트용)
        #    창 좌측 위에 얹어서 누르면 스트리밍이 시작되도록 함
        btn_frame = tk.Frame(self.root)
        btn_frame.place(x=10, y=680)  # 창 맨 아래 쪽에 버튼
        self.start_btn = tk.Button(btn_frame, text="스트림 시작", command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = tk.Button(btn_frame, text="스트림 중지", command=self.stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)


    def _schedule_video_update(self):
        """
        약 30ms마다 큐에서 최신 프레임을 꺼내와 Label에 표시
        """
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            self.photo_image = ImageTk.PhotoImage(image=pil_img)
            self.video_label.config(image=self.photo_image)

        self.root.after(30, self._schedule_video_update)

    def update_warning(self, text: str):
        """
        MetadataThread가 전달한 경고문(text)을 warning_label에 표시.
        빈 문자열("")이 들어오면 레이블이 사라집니다.
        """
        self.warning_label.config(text=text)

    def start_stream(self):
        """
        스트리밍 스레드가 실행 중이 아니면 새로 시작합니다.
        """
        if self.stream_thread and self.stream_thread.running:
            return

        print("[App] 스트리밍 스레드 시작 요청됨.")
        self.stream_thread = StreamThread(frame_queue=self.frame_queue)
        self.stream_thread.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_stream(self):
        """
        스트리밍 스레드를 중지하고 버튼 상태를 복원합니다.
        """
        if self.stream_thread and self.stream_thread.running:
            print("[App] 스트리밍 스레드 종료 요청됨.")
            self.stream_thread.running = False
            self.stream_thread.join()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


    def on_close(self):
        """
        프로그램 종료 시 스트리밍 스레드를 안전히 중단하고 창을 닫습니다.
        """
        print("[App] 종료 요청됨. 스레드 정리 중…")
        if self.stream_thread and self.stream_thread.running:
            self.stream_thread.running = False
            self.stream_thread.join()
        self.root.destroy()

# ======================================
# 프로그램 진입점
# ======================================

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
