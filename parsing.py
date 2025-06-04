import requests
import os
import time
from datetime import datetime
import struct
import cv2
from playsound import playsound
from gtts import gTTS

# 설정
CAM_IP = "169.254.248.57"
SAVE_DIR = "./captures"
VOICE_DIR = "./tts_voice"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)

# classid 해석 (C# 기준과 동일)
def get_label_name(classid: int):
    if classid == 1:
        return "No Helmet"
    elif classid == 2:
        return "With Helmet"
    else:
        return "Two Person"

# 음성 문구
SPEECH_TEXT = {
    "No Helmet": "헬멧을 쓰세요!",
    "Two Person": "한 명만 탑승해주세요!"
}

# 클래스 색상
LABEL_COLORS = {
    "No Helmet": (0, 0, 255),
    "With Helmet": (0, 255, 0),
    "Two Person": (255, 255, 0)
}

# 이미지 다운로드
def http_get_image(ip: str, save_path: str, user: str = "root", password: str = "root", timeout_sec: int = 5) -> bool:
    try:
        url = f"http://{ip}/cgi-bin/fwcamimg.cgi?FwModId=0&PortId=3&FwCgiVer=0x0001"
        response = requests.get(url, auth=(user, password), timeout=timeout_sec, stream=True)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"[ERROR] HTTP Response {response.status_code}")
            return False
    except Exception as e:
        print(f"[EXCEPTION] Failed to get image: {e}")
        return False

# 메타데이터 파싱
def parse_ai_objects(ai_data: bytes):
    objs = []
    for i in range(0, len(ai_data), 12):
        chunk = ai_data[i:i+12]
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

# JPG에서 메타데이터 추출
def get_ai_info(filename: str):
    try:
        with open(filename, 'rb') as f:
            read_data = f.read()

        if len(read_data) < 318:
            return []

        n_file_size = (read_data[48] << 24) | (read_data[49] << 16) | (read_data[50] << 8) | read_data[51]
        actual_size = len(read_data)

        if n_file_size != actual_size:
            return []

        ai_data = read_data[86:86+232]
        return parse_ai_objects(ai_data)

    except Exception as e:
        print(f"[EXCEPTION] Get AI Info 에러: {e}")
        return []

# 바운딩박스 + 라벨
def draw_boxes(image_path, objects, scale=0.25):
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))

    for obj in objects:
        x = int(obj['x'] / 256 * orig_w * scale)
        y = int(obj['y'] / 256 * orig_h * scale)
        bw = int(obj['w'] / 256 * orig_w * scale)
        bh = int(obj['h'] / 256 * orig_h * scale)

        label = get_label_name(obj['classid'])
        color = LABEL_COLORS.get(label, (255, 255, 255))
        label_text = f"{label} ({obj['confidence']}%)"

        cv2.rectangle(resized, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(resized, label_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return resized

# 음성 출력 (자동 생성 포함)
def speak(label: str):
    if label not in SPEECH_TEXT:
        return

    text = SPEECH_TEXT[label]
    mp3_path = os.path.join(VOICE_DIR, f"{label}.mp3")

    if not os.path.exists(mp3_path):
        try:
            tts = gTTS(text=text, lang='ko')
            tts.save(mp3_path)
        except Exception as e:
            print(f"[TTS 생성 실패] {label}: {e}")
            return

    try:
        playsound(mp3_path)
    except Exception as e:
        print(f"[음성 재생 오류] {label}: {e}")

# 실시간 루프
def live_stream():
    print("스트리밍 시작 (ESC 누르면 종료)")
    while True:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")

        success = http_get_image(CAM_IP, filename)
        if not success:
            print("이미지 다운로드 실패")
            time.sleep(1.0)
            continue

        objs = get_ai_info(filename)
        print(f"객체 수: {len(objs)}")

        labels_this_frame = set()
        for idx, obj in enumerate(objs):
            label = get_label_name(obj['classid'])

            print(f"[{idx}] {label} | 신뢰도: {obj['confidence']}% | "
                  f"x:{obj['x']} y:{obj['y']} w:{obj['w']} h:{obj['h']}")
            labels_this_frame.add(label)

        for label in labels_this_frame:
            speak(label)

        img = draw_boxes(filename, objs, scale=0.25)
        cv2.imshow("AI Camera", img)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(1.0)
        os.remove(filename)

    cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    live_stream()
