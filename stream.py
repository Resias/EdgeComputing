import threading
import requests
import struct
import time
import os
from datetime import datetime

import cv2
import numpy as np

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from urllib.parse import urlparse, urlunparse


# ======================================
# 설정 부분
# ======================================

# RTSP 스트림 URL (반드시 정확히 입력)
CAM_IP = "192.168.0.100"
RTSP_URL = f"rtsp://root:root@{CAM_IP}:554/cam0_0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# HTTP Snapshot용 IP (JPEG 메타데이터 파싱에만 사용)

SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def mask_rtsp_url(url: str) -> str:
    """
    rtsp://user:pass@host:port/path → rtsp://user:****@host:port/path
    """
    p = urlparse(url)
    user = p.username or ""
    # 사용자 이름은 남기고, 비밀번호는 **** 로 마스킹
    masked_userinfo = f"{user}:****"
    # 호스트·포트는 전부 *** 로 대체
    masked_netloc = f"{masked_userinfo}@***"
    # 경로도 보안 상 필요 없으면 *** 으로 가릴 수 있습니다
    masked_path = "/***" if p.path else ""
    return urlunparse((p.scheme, masked_netloc, masked_path, "", "", ""))

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
        x, y, w, h, confidence, classid = struct.unpack('<BBBBBB', chunk[:6])
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

def get_ai_info(filename: str, log_callback):
    """
    JPEG 바이트 전체(인코딩된 프레임)에서 AI 메타데이터(232바이트 영역)만 추출해 parse_ai_objects() 호출.
    - JPEG 헤더 offset 48~51에서 파일 크기 정보를 읽어 실제 길이와 비교
    - offset 86부터 232바이트 구간을 parse_ai_objects()에 전달
    """
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        # 파일 길이 체크
        if len(data) < 318:
            return []

        # 헤더 48~51: big-endian file size
        reported_size = (data[48] << 24) | (data[49] << 16) | (data[50] << 8) | data[51]
        actual_size = len(data)
        if reported_size != actual_size:
            # C# 쪽에서 에러 처리한 부분과 동일
            log_callback(f"JPG size mismatch: header={reported_size}, actual={actual_size}","ERROR")
            # print(f"[ERROR] JPG size mismatch: header={reported_size}, actual={actual_size}")
            return []

        # AI 메타데이터 블록 추출 (86~86+232)
        ai_block = data[86 : 86 + 232]
        return parse_ai_objects(ai_block)
    
    except Exception as e:
        log_callback(f"Get AI Info 에러: {e}","ERROR")
        # print(f"[EXCEPTION] Get AI Info 에러: {e}")
        return []

# ======================================
# HTTP로 JPEG Snapshot 가져오기
# ======================================

def http_get_image(ip: str, save_path: str, log,
                   user: str = "root", password: str = "root",
                   timeout_sec: int = 5) -> bool:
    """
    HTTP Snapshot URL에서 JPEG을 받아서 save_path에 저장.
    """
    try:
        url = f"http://{ip}/cgi-bin/fwcamimg.cgi?FwModId=0&PortId=3&FwCgiVer=0x0001"
        response = requests.get(url, auth=(user, password), timeout=timeout_sec, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            log(f"HTTP Snapshot Response {response.status_code}", "ERROR")
            # print(f"[ERROR] HTTP Snapshot Response {response.status_code}")
            return False
    except Exception as e:
        log(f"Failed to fetch JPEG: {e}", "ERROR")
        # print(f"[EXCEPTION] Failed to fetch JPEG: {e}")
        return False

# ======================================
# RTSP 스트리밍 전용 스레드
# ======================================

class StreamThread(threading.Thread):
    def __init__(self, update_callback, log_callback):
        super().__init__(daemon=True)
        self.update_callback = update_callback  # PIL 이미지를 GUI에 표시할 콜백
        self.log = log_callback
        self.running = False
        self.cap = None

    def run(self):
        # 1) RTSP 스트림 열기
        masked = mask_rtsp_url(RTSP_URL)
        self.log(f"RTSP 스트림 열기 시도: {masked}", "INFO")
        self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            # print(f"[ERROR] RTSP 스트림을 열 수 없습니다: {RTSP_URL}")
            self.log(f"RTSP 연결 실패: {masked}", "ERROR")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.log("프레임 읽기 실패, 재시도 중", "WARN")
                # 스트림 읽기에 실패하면 잠시 대기 후 재시도
                time.sleep(0.005)
                continue

            # 2) 1280×640으로 리사이즈 (GUI 상단 영역 크기)
            display_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LANCZOS4)
            # 3) BGR → RGB → PIL Image 변환
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # 4) GUI에 전달하여 video_label 갱신
            self.update_callback(pil_img)

            # 5) 아주 짧게 대기 (너무 짧으면 CPU 과부하, 길면 화면 끊김)
            time.sleep(0.01)

        self.cap.release()
        self.log("StreamThread 종료", "INFO")

# ======================================
# 메타데이터 파싱 전용 스레드
# ======================================

class MetadataThread(threading.Thread):
    def __init__(self, update_callback, log_callback, interval: float = 30.0):
        super().__init__(daemon=True)
        self.update_callback = update_callback  # 경고문 문자열을 GUI에 표시할 콜백
        self.log = log_callback
        self.interval = interval  # 대기 시간(초)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            # 1) HTTP Snapshot으로 JPEG 바이트 가져오기
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            fname = os.path.join(SNAPSHOT_DIR, f"shot_{ts}.jpg")
            self.log(f"HTTP 스냅샷 요청: {fname}", "DEBUG")
            if not http_get_image(CAM_IP, fname, log=self.log):
                self.log(f"스냅샷 요청 실패: {CAM_IP}", "ERROR")
                time.sleep(self.interval)
                continue
            # 2) 저장된 JPEG에서 AI 정보 파싱
            self.log(f"스냅샷 저장 완료: {fname}", "INFO")
            objs = get_ai_info(fname, self.log)
            len(objs)
            # for obj in objs:
            #     print(f"classid={obj['classid']}, conf={obj['confidence']}")

            self.update_callback(objs)
            
            # 4) 임시 파일 삭제
            try:
                os.remove(fname)
            except OSError:
                pass

            # 30초 동안 1초마다 종료 체크
            for _ in range(int(self.interval)):
                if not self.running:
                    break
                time.sleep(1)

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

        # (1) 비디오 표시
        self.video_label = tk.Label(self.root)
        self.video_label.place(x=0, y=0, width=640, height=640)
        # (2) 냉방 아이콘
        self.cooling_img_label = tk.Label(self.root, bg="white", relief="solid", bd=2)
        self.cooling_img_label.place(x=640, y=0, width=640, height=540)
        # (3) 로그 창
        self.log_widget = ScrolledText(root, state='disabled', font=("맑은 고딕",12))
        self.log_widget.place(x=640, y=540, width=640, height=120)

        self.ai_widget = ScrolledText(root, state='disabled', font=("맑은 고딕",12))
        self.ai_widget.place(x=640, y=660, width=640, height=60)

        
        self.last_person_time = datetime.now()
        self.no_person_logged = False

        # 스레드 및 이미지 저장 변수
        self.cooling_photo = None  # 이미지 객체 저장
        self.photo_image = None
        self.stream_thread = None
        self.meta_thread = None


        # (3) 버튼 프레임: 하단 y=650 위치
        btn_frame = tk.Frame(self.root)
        btn_frame.place(x=20, y=650)

        self.start_btn = tk.Button(btn_frame, text="시작(Start)", width=12, command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="정지(Stop)", width=12, command=self.stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] [{level}] {msg}"
        # print(full)               # 터미널 출력용
        self.log_widget.config(state='normal')
        self.log_widget.insert(tk.END, full + "\n")
        self.log_widget.config(state='disabled')

    def update_video(self, pil_image):
        """
        StreamThread가 전달한 PIL 이미지를 PhotoImage로 변환 후
        video_label에 표시.
        """
        try:
            self.photo_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=self.photo_image)
            self.video_label.image = self.photo_image
        except tk.TclError:
            # 윈도우가 이미 닫혔으면 무시
            pass
    
    def log_ai_info(self, objs):
        try:
            self.ai_widget.config(state='normal')
            self.ai_widget.delete('1.0', tk.END)
            cnt = get_human(objs)
            if cnt>0:
                line = (
                    # f"\n[{i}] classid={obj['classid']}, confidence={obj['confidence']} | "
                    # f"x={obj['x']} y={obj['y']} w={obj['w']} h={obj['h']}\n"
                    f"{WARNING_TEXT['Person']}"
                    f"\n현재 이용중인 인원: {cnt}"
                )
                self.ai_widget.insert(tk.END, line)
                self.ai_widget.config(state='disabled')
            else:
                self.ai_widget.insert(tk.END, WARNING_TEXT["No Person"])
                self.ai_widget.config(state='disabled')
        except tk.TclError:
            pass

    def update_icons(self, text: str, people_count: int):
        try:
            # 사람 수에 따라 이미지 결정
            if 2 < people_count <= 3:
                icon_path = os.path.join(
                    BASE_DIR, 
                    "icons",               # 실제 icons 폴더가 stream.py 파일과 같은 폴더에 있다면
                    "cooling_high.png"      # 또는 상대경로에 맞춰 조정
                )
                dynamic_msg = "High occupancy detected. Activating maximum cooling mode (18°C)."
            elif 1 < people_count <= 2:
                icon_path = os.path.join(
                    BASE_DIR, 
                    "icons",               # 실제 icons 폴더가 stream.py 파일과 같은 폴더에 있다면
                    "cooling_medium.png"      # 또는 상대경로에 맞춰 조정
                )
                dynamic_msg = "Moderate crowd detected. Cooling adjusted to strong mode (20°C)."
            elif people_count == 1:
                icon_path = os.path.join(
                    BASE_DIR, 
                    "icons",               # 실제 icons 폴더가 stream.py 파일과 같은 폴더에 있다면
                    "cooling_low.png"      # 또는 상대경로에 맞춰 조정
                )
                dynamic_msg = "No more than 10 people were detected. The air conditioner switches to minimum cooling."
            else:
                icon_path = os.path.join(
                    BASE_DIR, 
                    "icons",               # 실제 icons 폴더가 stream.py 파일과 같은 폴더에 있다면
                    "cooling_zero.png"      # 또는 상대경로에 맞춰 조정
                )
                dynamic_msg = "No one in Classroom."

            # 이미지 불러오기 및 표시
            try:
                self.log(dynamic_msg, "INFO")
                img = Image.open(icon_path).resize((300, 300))  # 적당한 크기로 조정
                self.cooling_photo = ImageTk.PhotoImage(img)
                self.cooling_img_label.config(image=self.cooling_photo)
                self.cooling_img_label.image = self.cooling_photo
            except Exception as e:
                self.log(f"아이콘 로딩 실패: {e}", "ERROR")
                # print(f"[ERROR] 아이콘 로딩 실패: {e}")
        except tk.TclError:
            pass


    def on_metadata(self, objs):
        # 로그와 경고문/아이콘 동시 갱신
        self.log_ai_info(objs)
        cnt = get_human(objs)
        now = datetime.now()
        
        if cnt > 0:
            # 사람이 탐지되면 시간 갱신, 로그 플래그 리셋
            self.last_person_time = now
            self.no_person_logged = False
        else:
            # 사람이 없으면 경과 시간 계산
            elapsed = (now - self.last_person_time).total_seconds()
            if elapsed >= 120 and not self.no_person_logged:
                self.ai_widget.insert(tk.END, "2분 동안 강의실 이용자 X\n")
                self.ai_widget.insert(tk.END, WARNING_TEXT["No Person"])
                self.ai_widget.config(state='disabled')
                self.no_person_logged = True
        txt = WARNING_TEXT["Person"] if cnt>0 else WARNING_TEXT["No Person"]
        self.update_icons(txt, cnt)

    def start_stream(self):
        """
        StreamThread와 MetadataThread를 생성·시작하고,
        버튼 상태를 전환합니다.
        """
        if self.stream_thread and self.stream_thread.is_alive():
            return

        # ① 스트리밍 스레드
        # update_video 대신, root.after로 안전하게 호출하도록 래핑
        self.stream_thread = StreamThread(
            update_callback=lambda img: self.root.after(0, self.update_video, img),
            log_callback=self.log
        )
        # self.stream_thread = StreamThread(update_callback=self.update_video) 
        self.stream_thread.start()

        # ② 메타데이터 파싱 스레드
        self.meta_thread = MetadataThread(
            update_callback=lambda objs: self.root.after(0, self.on_metadata, objs),
            interval=5.0,
            log_callback=self.log
        )
        self.meta_thread.start()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)


    def stop_stream(self):
        """
        실행 중인 두 스레드를 중단하고, 버튼 상태를 원래대로 복원합니다.
        """
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.running = False
        if self.meta_thread and self.meta_thread.is_alive():
            self.meta_thread.running = False

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        # 경고문 제거
        # self.update_warning("", 0)

    def on_close(self):
        """
        창을 닫을 때 두 스레드를 안전히 중단한 뒤 Tkinter 루프 종료.
        """
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.running = False
            # self.stream_thread.join()

        if self.meta_thread and self.meta_thread.is_alive():
            self.meta_thread.running = False
            # self.meta_thread.join()

        self.root.destroy()

# ======================================
# 프로그램 진입점
# ======================================

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
