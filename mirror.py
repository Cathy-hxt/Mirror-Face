import cv2
import os
import json
import numpy as np
import datetime
import atexit
from deepface import DeepFace
import time
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import Counter
import atexit
import io
from PIL import Image
from qianfan import Qianfan
from qianfan.resources.console.iam import IAM
from openai import OpenAI
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
import matplotlib.animation as animation
import screeninfo
from screeninfo import get_monitors
import threading

# ----------------- 人脸检测的时间变量 -----------------
face_present = False
face_start_time = None
face_end_time = None
recognizing = False
# ------------------Face++ API Key-------------------
FACEPP_API_KEY = "qWTsVPmDQmwxnpveEbcAt0FGUklv6bBW"
FACEPP_API_SECRET = "abUpRXtULBRMT6PxkyouaO7OnEt9tP6v"

# ------------------百度千帆 API Key-------------------
AccessKeyID = "ALTAKZyDAaT3UU2MudxB2u5Xzn"
AccessKeySecret = "7149e58678a74c87b5502f44ae667dd9"

# ----------------- 文件路径 -----------------
EMBEDDINGS_FILE = r"E:\emotion_db\face_embeddings.npz"
IDS_FILE = r"E:\emotion_db\face_ids.json"
LOGS_FILE = r"E:\emotion_db\emotion_logs.json"

# ----------------- 人脸库路径 -----------------
FACE_DB_PATH = r"E:\emotion_db\face_db"  # 你要保证这个文件夹存在，里面存图片用作DeepFace.find数据库

# ----------------- 加载持久化数据 -----------------
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

# ----------------- 保存数据函数 -----------------
def save_state():
    if known_face_embeddings:
        np.savez(EMBEDDINGS_FILE, *known_face_embeddings)
    with open(IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(known_face_ids, f, ensure_ascii=False, indent=4)
    with open(LOGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(emotion_logs, f, ensure_ascii=False, indent=4)

atexit.register(save_state)

# ----------------- Face++人脸检测函数 -----------------
def facepp_detect(FACEPP_API_KEY , FACEPP_API_SECRET, face_img):
    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    # 编码为 jpg 并获取字节流
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
        "return_attributes": "emotion,gender,age,beauty",  # 需检测的属性（情绪、性别、年龄、颜值）
        "return_landmark": 0  # 不检测关键点，可选 0/1/2
    }
    
    try:
        response = requests.post(url, files=files, params=params)
        result = response.json()
        return result
    except Exception as e:
        print(f"请求失败：{str(e)}")
        return None

# ----------------- OpenCV 人脸检测 -----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ----------------- AI对话API配置 -----------------
client = OpenAI(
    api_key="bce-v3/ALTAK-6IRSzgWrWUIuKxDGDu7J4/967e7d35e7d53c496a793dff4d5834e3e911e850;ZjkyZmQ2YmQxZTQ3NDcyNjk0ZTg1ZjYyYjlkZjNjODB8AAAAAAYCAADdZDVD4YcHu+BqhbfHLlyBSftws7uRNFuTATBl/DPV2ZsKjHS9pyMw8Nfh/uGCe5a5vhHRLrxDv2p1+GRq3AKBdbBqLhfS3nCZ2MEYcUqg3TBbeSgjwPIVwHaDXFLDxQwhLnIXRyv9dpYhNwzM5TLeJvb7PD/fP4UfunUu26kayyzuG853AfWQFMmv3a7QxgWfiMxGFIpRIZHATMxfa93MXiDVBDhW0tJUpjjYXY8qSesLuvAbhoNVf7H4W29n3rJ8Bjt9kvWtVquF8vFV/joE2pAeklYdWzsFJ7tnX0IInBQdYyp5B58se2jp7QypUrQIa9f8Z90ErHqspasu4oi2z0grfeTLUcAHod0sdes4yRIaN5yn+FeeXGjG7/mOCnskIFr7v+fcL5db/+9qw9qQO/GqWf5UuGcTKKVVWPA+nzJQVPPXxqnAbLPB+uRhwqE=",
    base_url="https://qianfan.baidubce.com/v2",
)

# ----------------- AI对话情绪表情映射 -----------------
emotion_emoji = {
        "happiness": "😀", "neutral": "😐", "sadness": "😞", "anger": "😡", "fear": "😨",
        "disgust": "😷", "surprise": "😱", "unknown": "❓", "": ""
}

def get_chat_response(name, emotion, gender, age, beauty_score):
    prompt = f"用户 {name} 当前情绪为 {emotion},性别：{gender},年龄：{age} 岁, 颜值：{beauty_score} 分(满分100,大于60就是属于是帅或美了)。请生成一句自然的对话内容，表达对用户情绪的理解或者安慰等。（不需要在对话中提及用户具体年龄，性别和颜值分数，不需要和用户继续发生对话。可以根据用户信息加上一个不要太油腻的称呼。 输出英文）"
    completion = client.chat.completions.create(
        model="ernie-3.5-8k",
        messages=[{'role': 'user', 'content': prompt}]
    )
    return completion.choices[0].message.content.strip()


def get_emotion_clock_img(emotion_logs, person_id):
    """
    生成艺术感更强的情绪时钟图片，返回OpenCV格式
    """

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 取今天的情绪日志
    hours = list(range(7, 19))  # 7:00-18:00 共12小时
    times = [f"{h}:00" for h in hours]
    today = datetime.datetime.now().date().isoformat()
    person_logs = [
        log for log in emotion_logs.get(person_id, [])
        if log['timestamp'][:10] == today
    ]
    # 归类到距离整点最近的小时
    hour_log_map = {}
    for log in person_logs:
        t = datetime.datetime.fromisoformat(log['timestamp'])
        min_diff = 60 * 60
        best_hour = None
        for hour in hours:
            dt_hour = t.replace(hour=hour, minute=0, second=0, microsecond=0)
            diff = abs((t - dt_hour).total_seconds())
            if diff < min_diff:
                min_diff = diff
                best_hour = hour
        if best_hour not in hour_log_map or min_diff < hour_log_map[best_hour][1]:
            hour_log_map[best_hour] = (log, min_diff)
    # 统计每小时情绪
    hour_emotion = []
    for hour in hours:
        if hour in hour_log_map:
            hour_emotion.append(hour_log_map[hour][0]['emotion'])
        else:
            hour_emotion.append("")

    emotions = [emotion_emoji.get(e, "") for e in hour_emotion]

    fig, ax = plt.subplots(figsize=(8,8), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_aspect('equal')

    n = 12
    radius = 1.0
    d = 0.2  # 竖直方向偏移

    theta0 = -2 * np.pi / 3  # -120度
    for i in range(n):
        theta = theta0 - i * np.pi / 6
        x0 = radius * np.cos(theta)
        y0 = radius * np.sin(theta)
        ax.text(x0, y0, emotions[i], fontsize=40, ha='center', va='center', color='white')
        ax.text(x0, y0 + d, times[i], fontsize=14, ha='center', va='bottom', color='white')

    # 时针指向当前小时
    now = datetime.datetime.now()
    # 超过半小时就进位到下一个小时
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

# ----------------- 显示对话和时钟的函数 -----------------
def show_dialog_clock_emotion(dialog_text, clock_img, most_common_emotion):
    global black_root
    if black_root:
        black_root.destroy()
        black_root = None
        
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
        root.after(2000, show_text)  # 保持2秒后显示文本

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
            # 1) 已完全显示的整行
            for idx in range(current_line_idx):
                line = text_lines[idx]
                bbox = draw.textbbox((0, 0), line, font=font_text)
                w_line = bbox[2] - bbox[0]
                h_line = bbox[3] - bbox[1]
                y = int(screen_h * 0.5) + idx * h_line
                draw.text(((screen_w - w_line) // 2, y), line, font=font_text, fill=(255,255,255))
            # 2) 当前行的部分文本
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


def init_black_screen():
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
    # 开一个后台线程运行主循环，不阻塞后面的逻辑
    threading.Thread(target=black_root.mainloop, daemon=True).start()

# ----------------- 初始化黑屏 -----------------
init_black_screen()

# ----------------- 打开摄像头 -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# ----------------- 初始化状态 -----------------
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

# ----------------- 人脸检测和识别主循环 -----------------
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
            # 人脸持续出现>2s且未开始识别，且未显示主情绪
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
                    clock_img = get_emotion_clock_img(emotion_logs, session_person_id)
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
            result = facepp_detect(FACEPP_API_KEY, FACEPP_API_SECRET, face_img) # 通过Face++ API进行情绪识别
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
