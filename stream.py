import threading
import requests
import struct
import time

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
    
def get_human(objs):
    """
    Filters and counts persons from object list based on classid==15.
    Returns: count (int)
    """
    return sum(1 for obj in objs if obj['classid'] == 15)

# 경고문 매핑
WARNING_TEXT = {
    "Person": "현재 이용중인 강의실입니다.",
    "No Person": "강의실이 비어있습니다."
}

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

# ======================================
# RTSP 스트리밍 전용 스레드
# ======================================

class StreamThread(threading.Thread):
    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback  # PIL 이미지를 GUI에 표시할 콜백
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
                time.sleep(0.05)
                continue

            # 2) 640×640으로 리사이즈 (윈도우 왼쪽 절반 영역)
            display_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LANCZOS4)




            # 3) BGR → RGB → PIL Image 변환
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # 4) GUI에 전달하여 video_label 갱신
            self.update_callback(pil_img)

            # 5) 아주 짧게 대기 (너무 짧으면 CPU 과부하, 길면 화면 끊김)
            time.sleep(0.01)

        # 루프 중단 시 RTSP 스트림 해제
        if self.cap:
            self.cap.release()

# ======================================
# 메타데이터 파싱 전용 스레드
# ======================================

class MetadataThread(threading.Thread):
    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback  # 경고문 문자열을 GUI에 표시할 콜백
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            # 1) HTTP Snapshot으로 JPEG 바이트 가져오기
            jpeg_bytes = fetch_jpeg_bytes(CAM_IP)
            warning_text = ""
            if jpeg_bytes:
                # 2) JPEG 바이트에서 AI 메타데이터 파싱
                objs = get_ai_info_from_bytes(jpeg_bytes)

                # 3) 파싱된 객체 리스트로부터 라벨 모음 생성
                labels = { get_label_name(obj['classid']) for obj in objs }

                # 4) 각 라벨에 대응하는 한글 경고문 합치기
                if labels:
                    messages = [WARNING_TEXT[label] for label in labels if label in WARNING_TEXT]
                    warning_text = "\n".join(messages)

            
                count = get_human(objs)
                warning_text = ""
                if labels:
                    messages = [WARNING_TEXT[label] for label in labels if label in WARNING_TEXT]
                    warning_text = "\n".join(messages)

            # GUI에 전달 (경고문 + 사람 수)
            self.update_callback(warning_text, count)


            time.sleep(1.0)




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

        # (1) 비디오 표시용 Label: 상단 왼쪽 절반(640×640)
        self.video_label = tk.Label(self.root)
        self.video_label.place(x=0, y=0, width=640, height=640)

        # (2) 경고문 Label: 비디오 레이블 위에 오버레이 (좌상단 20,20)
        self.warning_label = tk.Label(
            self.root,
            text="",
            font=("맑은 고딕", 24, "bold"),
            fg="red"
            # bg 옵션 제거
        )
        self.warning_label.place(x=20, y=20)

        # (3) 버튼 프레임: 하단 y=650 위치
        btn_frame = tk.Frame(self.root)
        btn_frame.place(x=20, y=650)

        self.start_btn = tk.Button(btn_frame, text="시작(Start)", width=12, command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="정지(Stop)", width=12, command=self.stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 스레드 및 이미지 저장 변수
        self.stream_thread = None
        # self.meta_thread = None
        self.meta_thread = MetadataThread(update_callback=lambda w, c: self.update_warning(w, c))

        self.photo_image = None  # 최신 프레임 저장용

        # (4) 냉방 아이콘 이미지 표시 라벨 (우측 상단 640x540)
        self.cooling_img_label = tk.Label(self.root, bg="white", relief="solid", bd=2)
        self.cooling_img_label.place(x=640, y=0, width=640, height=540)
        self.cooling_photo = None  # 이미지 객체 저장



    def update_video(self, pil_image):
        """
        StreamThread가 전달한 PIL 이미지를 PhotoImage로 변환 후
        video_label에 표시.
        """
        self.photo_image = ImageTk.PhotoImage(image=pil_image)
        self.video_label.config(image=self.photo_image)

    def update_warning(self, text: str, people_count: int):
        self.warning_label.config(text=text)

        # 사람 수에 따라 이미지 결정
        if 20 < people_count < 30:
            icon_path = "EdgeComputing/icons/cooling_high.png"
            dynamic_msg = "High occupancy detected. Activating maximum cooling mode (18°C)."
        elif 10 < people_count <= 20:
            icon_path = "EdgeComputing/icons/cooling_medium.png"
            dynamic_msg = "Moderate crowd detected. Cooling adjusted to strong mode (20°C)."
        else:
            icon_path = "EdgeComputing/icons/cooling_low.png"
            dynamic_msg = "No more than 10 people were detected. The air conditioner switches to minimum cooling."

        # 이미지 불러오기 및 표시
        try:
            img = Image.open(icon_path).resize((300, 300))  # 적당한 크기로 조정
            self.cooling_photo = ImageTk.PhotoImage(img)
            self.cooling_img_label.config(image=self.cooling_photo)
            self.warning_label.config(text=f"{text}\n\n{dynamic_msg}")
        except Exception as e:
            print(f"[ERROR] 아이콘 로딩 실패: {e}")


    def start_stream(self):
        """
        StreamThread와 MetadataThread를 생성·시작하고,
        버튼 상태를 전환합니다.
        """
        if self.stream_thread and self.stream_thread.is_alive():
            return

        # ① 스트리밍 스레드
        self.stream_thread = StreamThread(update_callback=self.update_video)
        self.stream_thread.start()

        # ② 메타데이터 파싱 스레드
        self.meta_thread = MetadataThread(update_callback=self.update_warning)
        self.meta_thread.start()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        


    def stop_stream(self):
        """
        실행 중인 두 스레드를 중단하고, 버튼 상태를 원래대로 복원합니다.
        """
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.running = False
            self.stream_thread.join()

        if self.meta_thread and self.meta_thread.is_alive():
            self.meta_thread.running = False
            self.meta_thread.join()

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        # 경고문 제거
        self.update_warning("", 0)

    def on_close(self):
        """
        창을 닫을 때 두 스레드를 안전히 중단한 뒤 Tkinter 루프 종료.
        """
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.running = False
            self.stream_thread.join()

        if self.meta_thread and self.meta_thread.is_alive():
            self.meta_thread.running = False
            self.meta_thread.join()

        self.root.destroy()

# ======================================
# 프로그램 진입점
# ======================================

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
