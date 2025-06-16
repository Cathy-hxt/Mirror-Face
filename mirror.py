import os
import json
import cv2
import numpy as np
import datetime
import atexit
import time
import requests
import textwrap
from collections import Counter
import io
import threading
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont, ImageTk
from screeninfo import get_monitors
from deepface import DeepFace
from openai import OpenAI
from qianfan.resources.console.iam import IAM

# ------------------ 常量配置 ---------------------
# Face++ API 配置
FACEPP_API_KEY = "qWTsV*************************"
FACEPP_API_SECRET = "abUp*************************"

# 百度千帆 API 配置
ACCESS_KEY_ID = "ALT*************************"
ACCESS_KEY_SECRET = "7149e*************************"

# 文件路径配置
EMBEDDINGS_FILE = r"E:\emotion_db\face_embeddings.npz"
IDS_FILE = r"E:\emotion_db\face_ids.json"
LOGS_FILE = r"E:\emotion_db\emotion_logs.json"
FACE_DB_PATH = r"E:\emotion_db\face_db"  # 存储人脸图片的数据库目录

# ------------------ 加载持久化数据 ---------------------
if os.path.exists(EMBEDDINGS_FILE):
    data = np.load(EMBEDDINGS_FILE)
    known_face_embeddings = [data[key] for key in data.files]
else:
    known_face_embeddings = []

if os.path.exists(IDS_FILE):
    with open(IDS_FILE, 'r', encoding='utf-8') as f:
        known_face_ids = json.load(f)
else:
    known_face_ids = []

if os.path.exists(LOGS_FILE):
    with open(LOGS_FILE, 'r', encoding='utf-8') as f:
        emotion_logs = json.load(f)
else:
    emotion_logs = {}

# ------------------ 保存数据 ---------------------
def save_state():
    """保存当前的情绪识别数据、ID和日志到文件。"""
    if known_face_embeddings:
        np.savez(EMBEDDINGS_FILE, *known_face_embeddings)
    with open(IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(known_face_ids, f, ensure_ascii=False, indent=4)
    with open(LOGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(emotion_logs, f, ensure_ascii=False, indent=4)

atexit.register(save_state)

# ------------------ Face++ API 人脸检测 ---------------------
def facepp_detect(face_img):
    """使用Face++ API进行人脸检测并返回情绪、性别、年龄等属性。"""
    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    ret, buf = cv2.imencode('.jpg', face_img)
    if not ret:
        print("图像编码失败")
        return None
    files = {
        "image_file": ("face.jpg", buf.tobytes(), "image/jpeg")
    }
    params = {
        "api_key": FACEPP_API_KEY,
        "api_secret": FACEPP_API_SECRET,
        "return_attributes": "emotion,gender,age,beauty",
        "return_landmark": 0  # 不返回面部关键点
    }
    
    try:
        response = requests.post(url, files=files, params=params)
        result = response.json()
        return result
    except Exception as e:
        print(f"请求失败：{str(e)}")
        return None

# ------------------ OpenCV 人脸检测 ---------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------ AI 对话 API 配置 ---------------------
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.openai.com")

# ------------------ 情绪表情映射 ---------------------
emotion_emoji = {
    "happiness": "😀", "neutral": "😐", "sadness": "😞", "anger": "😡", "fear": "😨",
    "disgust": "😷", "surprise": "😱", "unknown": "❓", "": ""
}

# ------------------ AI 对话生成 ---------------------
def get_chat_response(name, emotion, gender, age, beauty_score):
    """根据用户的情绪信息生成对话内容。"""
    prompt = f"User {name} is feeling {emotion}, gender: {gender}, *************************."
    completion = client.chat.completions.create(
        model="*************",
        messages=[{'role': 'user', 'content': prompt}]
    )
    return completion.choices[0].message.content.strip()

# ------------------ 情绪时钟生成 ---------------------
def get_emotion_clock_img(person_id):
    """为指定的用户生成情绪时钟图像。"""
    today = datetime.datetime.now().date().isoformat()
    person_logs = [log for log in emotion_logs.get(person_id, []) if log['timestamp'][:10] == today]
    hour_emotion = [""] * 12  # 7 AM 到 6 PM，共12小时
    hours = list(range(7, 19))

    for log in person_logs:
        timestamp = datetime.datetime.fromisoformat(log['timestamp'])
        best_hour = min(hours, key=lambda hour: abs((timestamp - timestamp.replace(hour=hour, minute=0)).total_seconds()))
        hour_emotion[best_hour - 7] = log['emotion']

    *************************
    
    print(f"情绪时钟已保存到 {output_path}")
    return img_cv

# ------------------ 运行显示情绪与对话 ---------------------
def show_dialog_clock_emotion(dialog_text, clock_img, most_common_emotion):
    """在全屏下依次显示情绪表情、对话内容、情绪时钟"""
    monitor = get_monitors()[1]
    screen_w, screen_h = monitor.width, monitor.height

    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{screen_w}x{screen_h}+{monitor.x}+{monitor.y}")
    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg='black', highlightthickness=0)
    canvas.pack(fill='both', expand=True)

    def show_emoji():
        
        *************************

    def show_text():
           
        *************************
           
    def show_clock():
        
        *************************
            
    def clear_screen():
        canvas.delete("all")
        canvas.configure(bg='black')
        root.after(1000, root.destroy)  # 1秒后关闭窗口

    root.after(500, show_emoji)  # 0.5秒后开始
    root.mainloop()

# ------------------ 启动黑屏 ---------------------
def init_black_screen():
    """初始化并启动黑屏窗口。"""
    global black_root, black_canvas
    monitor = get_monitors()[1]
    black_root = tk.Tk()
    black_root.overrideredirect(True)
    black_root.geometry(f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}")
    black_canvas = tk.Canvas(black_root,
                             width=monitor.width,
                             height=monitor.height,
                             bg='black',
                             highlightthickness=0)
    black_canvas.pack(fill='both', expand=True)
    # 开启后台线程运行主循环，不阻塞后续逻辑
    threading.Thread(target=black_root.mainloop, daemon=True).start()
    
# 启动黑屏
init_black_screen()

# ------------------ 打开摄像头 ---------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# ------------------ 初始化状态 ---------------------
face_present = False
face_start_time = None
face_end_time = None
recognizing = False
main_emotion_displayed = False
dialog_displaying = False
dialog_text = ""
dialog_start_time = None
clock_img = None
most_common_emotion = None
destroyWindow = False
black_root = None

# ------------------ 人脸检测和识别主循环 ---------------------
while True:

    *************************

cap.release()
cv2.destroyAllWindows()
save_state()
print("程序结束，已保存所有数据")
