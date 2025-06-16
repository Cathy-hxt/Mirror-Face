import os
import json
import cv2
import numpy as np
import datetime
import atexit
import time
import requests
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
FACEPP_API_KEY = "qWTsVPmDQmwxnpveEbcAt0FGUklv6bBW"
FACEPP_API_SECRET = "abUpRXtULBRMT6PxkyouaO7OnEt9tP6v"

# 百度千帆 API 配置
ACCESS_KEY_ID = "ALTAKZyDAaT3UU2MudxB2u5Xzn"
ACCESS_KEY_SECRET = "7149e58678a74c87b5502f44ae667dd9"

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
    prompt = f"User {name} is feeling {emotion}, gender: {gender}, age: {age} years, beauty score: {beauty_score} out of 100. Please generate a natural conversation to express understanding or comfort."
    completion = client.chat.completions.create(
        model="ernie-3.5-8k",
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

    emotions = [emotion_emoji.get(e, "") for e in hour_emotion]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_aspect('equal')
    n = 12
    radius = 1.0
    d = 0.2
    theta0 = -2 * np.pi / 3  # -120度
    for i in range(n):
        theta = theta0 - i * np.pi / 6
        x0 = radius * np.cos(theta)
        y0 = radius * np.sin(theta)
        ax.text(x0, y0, emotions[i], fontsize=40, ha='center', va='center', color='white')
        ax.text(x0, y0 + d, str(hours[i]), fontsize=14, ha='center', va='bottom', color='white')

    # 时针指向当前小时
    now = datetime.datetime.now()
    hour = now.hour
    if now.minute >= 30:
        hour += 1
    hour_idx = hour - 7
    if hour_idx < 0:
        hour_idx = 0
    elif hour_idx >= n:
        hour_idx = n - 1
    hour_hand_angle = theta0 - hour_idx * np.pi / 6

    x_start, y_start = 0, 0
    x_end = 0.8 * np.cos(hour_hand_angle)
    y_end = 0.8 * np.sin(hour_hand_angle)

    arrow = FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle='-|>,head_width=0.15,head_length=0.25',
        color='#FF3366',
        linewidth=4,
        alpha=0.85,
        mutation_scale=30
    )
    ax.add_patch(arrow)

    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)
    ax.set_title(f'{person_id} Emotion Clock', fontsize=18, color='white', pad=20)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0) 

    plt.close(fig)
    buf.seek(0)
    img_pil = Image.open(buf).convert('RGB')
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # 保存到当前目录
    output_path = "E:\output.png"
    cv2.imwrite(output_path, img_cv)
    print(f"情绪时钟已保存到 {output_path}")
    return img_cv

# ------------------ 运行显示情绪与对话 ---------------------
def show_dialog_clock_emotion(dialog_text, clock_img, most_common_emotion):
    """在全屏显示情绪表情和对话内容。"""
    monitor = get_monitors()[1]
    screen_w, screen_h = monitor.width, monitor.height

    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{screen_w}x{screen_h}+{monitor.x}+{monitor.y}")
    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg='black', highlightthickness=0)
    canvas.pack(fill='both', expand=True)

    def show_emoji():
        canvas.delete("all")
        img = Image.new('RGB', (screen_w, screen_h), color='black')
        draw = ImageDraw.Draw(img)
        try:
            font_emoji = ImageFont.truetype("seguiemj.ttf", int(screen_h * 0.3))
        except:
            font_emoji = ImageFont.load_default()
        emoji = emotion_emoji.get(most_common_emotion, "")
        bbox = draw.textbbox((0, 0), emoji, font=font_emoji)
        w_emoji = bbox[2] - bbox[0]
        h_emoji = bbox[3] - bbox[1]
        draw.text(((screen_w-w_emoji)//2, int((screen_h-h_emoji)*0.5)), emoji, font=font_emoji, fill=(255,255,255))
        tk_img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor='nw', image=tk_img)
        canvas.image = tk_img
        root.after(2000, show_text)

    def show_text():
        canvas.delete("all")
        try:
            font_text = ImageFont.truetype("simhei.ttf", int(screen_h * 0.04))
        except:
            font_text = ImageFont.load_default()
        text_lines = textwrap.wrap(dialog_text, width=30)
        current_line_idx = 0
        current_char_idx = 0

        def update_text():
            nonlocal current_line_idx, current_char_idx
            canvas.delete("all")
            img = Image.new('RGB', (screen_w, screen_h), color='black')
            draw = ImageDraw.Draw(img)
            # 已完全显示的整行
            for idx in range(current_line_idx):
                line = text_lines[idx]
                bbox = draw.textbbox((0, 0), line, font=font_text)
                w_line = bbox[2] - bbox[0]
                h_line = bbox[3] - bbox[1]
                y = int(screen_h * 0.5) + idx * h_line
                draw.text(((screen_w - w_line) // 2, y), line, font=font_text, fill=(255,255,255))
            # 当前行的部分文本
            if current_line_idx < len(text_lines):
                line = text_lines[current_line_idx]
                partial = line[:current_char_idx+1]
                bbox = draw.textbbox((0, 0), partial, font=font_text)
                w_line = bbox[2] - bbox[0]
                h_line = bbox[3] - bbox[1]
                y = int(screen_h * 0.5) + current_line_idx * h_line
                draw.text(((screen_w - w_line) // 2, y), partial, font=font_text, fill=(255,255,255))
            # 更新画布
            tk_img = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor='nw', image=tk_img)
            canvas.image = tk_img
            # 更新索引并设定下一次延迟
            if current_line_idx < len(text_lines):
                current_char_idx += 1
                if current_char_idx >= len(text_lines[current_line_idx]):
                    current_char_idx = 0
                    current_line_idx += 1
                    delay = 500   # 行尾稍作停顿
                else:
                    delay = 100
                root.after(delay, update_text)
            else:
                root.after(2000, show_clock)

        update_text()
    def show_clock():
        canvas.delete("all")
        if clock_img is None:
            root.after(2000, clear_screen)
            return

        # 准备原始 PIL 图像
        if isinstance(clock_img, np.ndarray):
            clock_pil = Image.fromarray(clock_img)
        else:
            clock_pil = clock_img

        # 计算最终显示大小，保持比例，不超过 60% 宽度或 60% 高度
        max_w = int(screen_w * 0.5)
        max_h = int(screen_h * 0.5)
        orig_w, orig_h = clock_pil.size
        base_scale = max(max_w / orig_w, max_h / orig_h, 1.0)
        target_w = int(orig_w * base_scale)
        target_h = int(orig_h * base_scale)

        steps = 20         # 动画帧数
        interval = 50      # 每帧延迟 (ms)

        def animate(step):
            canvas.delete("all")
            factor = (step + 1) / steps
            w = int(target_w * factor)
            h = int(target_h * factor)
            frame = clock_pil.resize((w, h), Image.LANCZOS)

            img = Image.new('RGB', (screen_w, screen_h), color='black')
            x = (screen_w - w) // 2
            y = (screen_h - h) // 2
            img.paste(frame, (x, y))

            tk_img = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor='nw', image=tk_img)
            canvas.image = tk_img

            if step + 1 < steps:
                root.after(interval, lambda: animate(step + 1))
            else:
                root.after(2000, clear_screen)

        animate(0)
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
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(faces) > 0:
        if not face_present:
            face_start_time = time.time()
            face_present = True
            main_emotion_displayed = False
            dialog_displaying = False
            dialog_text = ""
            dialog_start_time = None
            clock_img = None
        else:
            # 人脸持续出现超过2秒并且未开始识别，且未显示主情绪
            if not recognizing and not main_emotion_displayed and time.time() - face_start_time >= 2:
                recognizing = True  # 满2秒，开始识别
                session_emotions = []  # 初始化情绪统计列表（人脸正对摄像头期间）
                gender = "unknown"
                age = "unknown"
                beauty_score = "unknown"
                session_start_time = None
                session_person_id = None
            # 满6秒，统计主情绪并显示对话和时钟
            if recognizing and not main_emotion_displayed and time.time() - face_start_time >= 6:
                recognizing = False # 人脸正对摄像头超过6秒,结束识别
                if session_emotions and session_person_id:
                    most_common_emotion = Counter(session_emotions).most_common(1)[0][0]
                    if session_person_id not in emotion_logs:
                        emotion_logs[session_person_id] = []
                    emotion_logs[session_person_id].append({
                        'timestamp': session_start_time,
                        'emotion': most_common_emotion,
                        'gender': gender,
                        'age': age,
                        'beauty_score': beauty_score
                    })
                    save_state()
                    # 获取今天的历史情绪
                    today = datetime.datetime.now().date().isoformat()
                    today_logs = [
                        log['emotion']
                        for log in emotion_logs[session_person_id]
                        if log['timestamp'][:10] == today
                    ]    
                    # 调用AI对话
                    conversation = get_chat_response(session_person_id, most_common_emotion, gender, age, beauty_score)
                    dialog_displaying = True
                    dialog_text = f"{conversation}"
                    dialog_start_time = time.time()                    
                    # 绘制时钟样式
                    clock_img = get_emotion_clock_img(session_person_id)
                    main_emotion_displayed = True  # 防止重复显示
        face_end_time = None  # 重置离开时间
    else:
        if face_present:
            if face_end_time is None:
                face_end_time = time.time()
            elif time.time() - face_end_time >= 3: # 人脸离开3秒，结束识别
                face_present = False
                recognizing = False  # 人脸离开后，结束识别
                face_start_time = None
                face_end_time = None
                main_emotion_displayed = False  # 重置主情绪显示状态
                dialog_displaying = False
                dialog_text = ""
                dialog_start_time = None
                clock_img = None
                destroyWindow = True  # 重置对话窗口状态
                most_common_emotion = None

    # 只有在recognizing为True时才进行识别
    if recognizing and len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = rgb_frame[y:y+h, x:x+w]

            # 用 DeepFace.find 在库里查找相似脸
            df_list = DeepFace.find(img_path=face_img, db_path=FACE_DB_PATH, 
                               model_name='Facenet', enforce_detection=False)
            df = df_list[0]  # 只取第一个结果
            if df.empty:
                # 库中无匹配，视为新面孔
                person_name = input("检测到新人，请输入名字：").strip()
                # 如果误识别为新人，按回车键跳过
                if not person_name:
                    print("已跳过本次未识别的人脸。")
                    continue  # 跳过本次循环

                # 新建新名字的文件夹
                person_dir = os.path.join(FACE_DB_PATH, person_name)
                os.makedirs(person_dir, exist_ok=True)

                # 采集多张图片
                num_samples = 3  # 你可以根据需要调整采集数量
                for i in range(num_samples):
                    # 重新读取一帧
                    ret, sample_frame = cap.read()
                    if not ret:
                        continue
                    rgb_sample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                    sample_face = rgb_sample[y:y+h, x:x+w]
                    save_path = os.path.join(person_dir, f"{person_name}_{i+1}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(sample_face, cv2.COLOR_RGB2BGR))
                    # 在窗口上显示采集进度
                    cv2.putText(
                        sample_frame,f"采集中: {i+1}/{num_samples}",(30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255),2
                    )
                    cv2.imshow('采集中', sample_frame)
                    cv2.waitKey(500)  # 每隔0.5秒采一张

                emotion_logs[person_name] = []
                save_state()
                person_id = person_name
            else:
                # 有匹配，选取相似距离最近的
                best_match = df.iloc[df['distance'].idxmin()]
                matched_img_path = best_match['identity']  # identity字段是图片绝对路径
                person_name = os.path.basename(os.path.dirname(matched_img_path))
                person_id = person_name

            # 情绪识别
            result = facepp_detect(face_img) # 通过Face++ API进行情绪识别
            if result and "faces" in result and result["face_num"] > 0:
                face = result["faces"][0]
                attributes = face["attributes"]
                # 提取情绪、性别、年龄和颜值分数
                if "emotion" in attributes:
                    emotions = attributes["emotion"]
                    # 取情绪分数最高的情绪
                    dominant_emotion = max(emotions, key=emotions.get)
                else:
                    dominant_emotion = "unknown" 
                if "gender" in attributes:
                    gender = attributes["gender"]["value"]
                if "age" in attributes:
                    age = attributes["age"]["value"]
                if "beauty" in attributes:
                    beauty_score = attributes["beauty"]["female_score"]
            ts = datetime.datetime.now().isoformat()
            if session_start_time is None:
                session_start_time = ts
                session_person_id = person_id
            session_emotions.append(dominant_emotion)
            if person_id not in emotion_logs:
                emotion_logs[person_id] = []

            # 视频显示
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{person_id}: {dominant_emotion}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    # 持续显示对话内容，直到人脸离开
    if dialog_displaying and dialog_text:
        show_dialog_clock_emotion(dialog_text, clock_img, most_common_emotion)
    if destroyWindow:
        cv2.destroyWindow('Dialog & Clock')
 
    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
save_state()
print("程序结束，已保存所有数据")
