"""
EOG -> ADB 手机屏幕控制 Demo
==============================
用分类结果控制安卓手机/模拟器的屏幕操作。

使用方法：
    # 键盘模拟测试（默认）
    python adb_control.py

    # 指定屏幕分辨率
    python adb_control.py --width 1080 --height 1920

前置条件：
    1. 安装 ADB 并加入 PATH
    2. USB 连接手机（开启 USB 调试）或启动安卓模拟器
    3. 运行 adb devices 确认设备已连接
"""

import subprocess
import time
import sys
import os
import numpy as np

# ============================================================
# 配置
# ============================================================
SCREEN_W = 1080   # 屏幕宽度（像素）
SCREEN_H = 1920   # 屏幕高度（像素）
COOLDOWN_SEC = 1.5 # 冷却时间（秒）
SWIPE_DURATION = 300  # 滑动持续时间（毫秒）
ADB_PATH = r"D:\Program Files\Netease\MuMu\nx_main\adb.exe"

# Live 模式配置
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
GAIN_H = 12.0
GAIN_V = 20.0
FILTER_BUFFER_SIZE = 150
WINDOW_SIZE = 50
CONFIDENCE_THRESHOLD = 0.7

# 虚拟光标配置
CURSOR_STEP = 80      # 每次移动的像素步长（手机坐标系）
CANVAS_SCALE = 4      # 缩放比例（手机像素 / 画布像素）

# 动作 -> ADB 命令映射（坐标在 send_action 中根据屏幕尺寸计算）
CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]


# ============================================================
# 核心：ADB 控制
# ============================================================
_last_action_time = {}


def check_adb():
    """检查 ADB 连接状态。"""
    try:
        result = subprocess.run(
            [ADB_PATH, "devices"], capture_output=True, text=True, timeout=5
        )
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        devices = [l for l in lines[1:] if 'device' in l and 'offline' not in l]
        if devices:
            print(f"  ADB 已连接 {len(devices)} 个设备:")
            for d in devices:
                print(f"    {d}")
            return True
        else:
            print("  [警告] 未检测到 ADB 设备")
            print("  请确认：USB 调试已开启 / 模拟器已启动 / adb devices 有输出")
            return False
    except FileNotFoundError:
        print("  [错误] 未找到 adb 命令，请安装 Android SDK Platform Tools 并加入 PATH")
        return False


def send_action(action, w=None, h=None):
    """
    根据分类结果发送 ADB 命令。

    参数：
        action: 分类名 ("Left"/"Right"/"Up"/"Down"/"Blink"/"Rest")
        w, h: 屏幕宽高（默认用全局配置）

    返回：True 如果命令已发送，False 如果被冷却跳过或无操作
    """
    if action == "Rest":
        return False

    # 冷却检查
    now = time.time()
    if action in _last_action_time:
        if now - _last_action_time[action] < COOLDOWN_SEC:
            return False

    w = w or SCREEN_W
    h = h or SCREEN_H
    cx, cy = w // 2, h // 2  # 屏幕中心
    margin_x = w * 3 // 8    # 水平滑动距离
    margin_y = h * 3 // 8    # 垂直滑动距离

    cmd = None
    desc = ""

    if action == "Left":
        cmd = f"input swipe {cx + margin_x} {cy} {cx - margin_x} {cy} {SWIPE_DURATION}"
        desc = "左划"
    elif action == "Right":
        cmd = f"input swipe {cx - margin_x} {cy} {cx + margin_x} {cy} {SWIPE_DURATION}"
        desc = "右划"
    elif action == "Up":
        cmd = f"input swipe {cx} {cy + margin_y} {cx} {cy - margin_y} {SWIPE_DURATION}"
        desc = "上划"
    elif action == "Down":
        cmd = f"input swipe {cx} {cy - margin_y} {cx} {cy + margin_y} {SWIPE_DURATION}"
        desc = "下划"
    elif action == "Blink":
        cmd = f"input tap {cx} {cy}"
        desc = "点击"

    if cmd:
        _last_action_time[action] = now
        print(f"  >> {desc} ({action}) -> adb shell {cmd}")
        subprocess.Popen([ADB_PATH, "shell", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    return False


def tap_at(x, y):
    """在指定坐标点击。"""
    cmd = f"input tap {x} {y}"
    print(f"  >> 点击 ({x}, {y}) -> adb shell {cmd}")
    subprocess.Popen([ADB_PATH, "shell", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def adb_key(keycode, desc=""):
    """发送安卓按键事件。"""
    cmd = f"input keyevent {keycode}"
    print(f"  >> {desc} -> adb shell {cmd}")
    subprocess.Popen([ADB_PATH, "shell", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ============================================================
# IMU 头控光标
# ============================================================
class HeadMouseController:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cursor_x = screen_w / 2
        self.cursor_y = screen_h / 2
        self.last_yaw = 0.0
        self.last_pitch = 0.0
        self.sensitivity_x = 25.0
        self.sensitivity_y = 25.0
        self.deadzone = 0.3
        self.is_initialized = False

    def recenter(self, yaw, pitch):
        self.last_yaw = yaw
        self.last_pitch = pitch
        self.cursor_x = self.screen_w / 2
        self.cursor_y = self.screen_h / 2
        self.is_initialized = True

    def update(self, yaw, pitch):
        if not self.is_initialized:
            self.recenter(yaw, pitch)
            return int(self.cursor_x), int(self.cursor_y)

        dy = yaw - self.last_yaw
        dp = pitch - self.last_pitch

        if abs(dy) < self.deadzone:
            dy = 0
        if abs(dp) < self.deadzone:
            dp = 0

        self.cursor_x += dy * self.sensitivity_x
        self.cursor_y += dp * self.sensitivity_y

        self.cursor_x = max(0, min(self.screen_w, self.cursor_x))
        self.cursor_y = max(0, min(self.screen_h, self.cursor_y))

        if dy != 0:
            self.last_yaw = yaw
        if dp != 0:
            self.last_pitch = pitch

        return int(self.cursor_x), int(self.cursor_y)


# ============================================================
# Live 模式：信号处理 + 模型推理
# ============================================================
def apply_filter(window_data):
    """带通滤波 0.5-10Hz。"""
    import scipy.signal as sig
    b, a = sig.butter(4, [0.5 / 25, 10.0 / 25], btype='band')
    out = np.zeros_like(window_data)
    out[:, 0] = sig.filtfilt(b, a, window_data[:, 0])
    out[:, 1] = sig.filtfilt(b, a, window_data[:, 1])
    return out


def extract_features(window):
    """提取 22 个特征（与训练一致）。"""
    from scipy.stats import skew, kurtosis
    from numpy.fft import rfft, rfftfreq
    features = []
    for axis in range(2):
        s = window[:, axis]
        d = np.diff(s)
        features.extend([
            np.std(s), np.max(s) - np.min(s),
            np.mean(np.abs(d)), np.max(np.abs(d)),
            skew(s), kurtosis(s), np.sum(s**2),
            np.sqrt(np.mean(s**2)),
        ])
        centered = s - np.mean(s)
        features.append(np.sum(np.diff(np.sign(centered)) != 0) / len(s))
        fft_v = np.abs(rfft(s))
        freqs = rfftfreq(len(s), d=1.0 / 50.0)
        features.append(np.max(fft_v[1:]) if len(fft_v) > 1 else 0.0)
        if len(fft_v) > 1 and np.sum(fft_v[1:]) > 0:
            features.append(np.sum(freqs[1:] * fft_v[1:]) / np.sum(fft_v[1:]))
        else:
            features.append(0.0)
    return np.array(features).reshape(1, -1)


def live_mode(port=None):
    """串口实时信号 -> 模型推理 -> ADB 控制。"""
    import serial
    import joblib
    from collections import deque

    port = port or SERIAL_PORT
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "rf_model.pkl")

    print(f"\n  加载模型: {model_path}")
    clf = joblib.load(model_path)

    print(f"  连接串口: {port} @ {BAUD_RATE}")
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    ser.reset_input_buffer()

    raw_buf = deque(maxlen=FILTER_BUFFER_SIZE)
    cooldown = 0
    print("  开始推理... (Ctrl+C 退出)\n")

    try:
        while True:
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        raw_buf.append([float(parts[0]) * GAIN_H,
                                        float(parts[1]) * GAIN_V])
                    except ValueError:
                        pass

            if len(raw_buf) >= FILTER_BUFFER_SIZE:
                long_win = np.array(list(raw_buf))[-FILTER_BUFFER_SIZE:]
                filtered = apply_filter(long_win)
                final_win = filtered[-WINDOW_SIZE:]

                feats = extract_features(final_win)
                probs = clf.predict_proba(feats)[0]
                pred_idx = np.argmax(probs)
                pred_label = CLASSES[pred_idx]
                conf = probs[pred_idx]

                if cooldown > 0:
                    cooldown -= 1
                elif pred_label != "Rest" and conf > CONFIDENCE_THRESHOLD:
                    sent = send_action(pred_label)
                    if sent:
                        print(f"  [{pred_label}] conf={conf:.0%}")
                        cooldown = 10
    except KeyboardInterrupt:
        print("\n  退出。")
    finally:
        ser.close()



def sim_mode():
    """模拟信号模式：生成假 EOG 信号 -> 模型推理 -> ADB 控制，测试整条链路。"""
    import joblib

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    clf = joblib.load(os.path.join(model_dir, "rf_model.pkl"))

    # 动作对应的信号方向: (H极性, V极性)
    action_map = {
        "Right": (1, 0), "Left": (-1, 0),
        "Up": (0, 1), "Down": (0, -1),
        "Blink": (0, 0),
    }

    print("\n" + "=" * 50)
    print("模拟信号模式 - 测试链路")
    print("=" * 50)
    print("  依次生成: Right -> Left -> Up -> Down -> Blink")
    print("  每个动作间隔 3 秒\n")

    for action_name, (h_pol, v_pol) in action_map.items():
        # 生成 50 点模拟信号
        t = np.linspace(0, 1, 50)
        wave = np.sin(2 * np.pi * 2 * t)
        wave[wave > 0] *= 0.6  # 模拟不对称
        amp = 180 + np.random.randn() * 20
        noise = np.random.randn(50) * 5

        h_sig = wave * amp * h_pol + noise
        v_sig = wave * amp * v_pol + noise
        if action_name == "Blink":
            v_sig = np.abs(np.sin(2 * np.pi * 3 * t)) * 300 + noise

        window = np.column_stack([h_sig, v_sig])
        feats = extract_features(window)
        probs = clf.predict_proba(feats)[0]
        pred_idx = np.argmax(probs)
        pred_label = CLASSES[pred_idx]
        conf = probs[pred_idx]

        match = "OK" if pred_label == action_name else "MISMATCH"
        print(f"  [{action_name}] -> 模型预测: {pred_label} ({conf:.0%}) {match}")
        send_action(pred_label)
        time.sleep(3)

    print("\n  模拟完成。")


# ============================================================
# 虚拟光标模式（截屏镜像 + 光标叠加）
# ============================================================
def grab_screen():
    """截取模拟器屏幕，返回 PIL Image 或 None。"""
    try:
        # 用较小分辨率截屏加快传输
        result = subprocess.run(
            [ADB_PATH, "exec-out", "screencap", "-p"],
            capture_output=True, timeout=3
        )
        if result.stdout:
            from io import BytesIO
            from PIL import Image
            img = Image.open(BytesIO(result.stdout))
            # 先缩到一半再传给主线程，减少后续处理开销
            half = (img.width // 2, img.height // 2)
            return img.resize(half, Image.BILINEAR)
    except Exception:
        pass
    return None


def cursor_mode():
    """截屏镜像 + 光标叠加的可视化控制模式。"""
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw
    import threading

    cw = SCREEN_W // CANVAS_SCALE
    ch = SCREEN_H // CANVAS_SCALE
    cursor_x = SCREEN_W // 2
    cursor_y = SCREEN_H // 2

    root = tk.Tk()
    root.title("EOG + IMU 虚拟光标")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=cw, height=ch, bg="#1a1a2e")
    canvas.pack()

    status_var = tk.StringVar(value="IJKL移动 / WASD翻页 / 空格点击 / R返回 / H主屏 / Q退出")
    tk.Label(root, textvariable=status_var, font=("Consolas", 10),
             bg="#16213e", fg="white", anchor="w", padx=5).pack(fill="x")

    coord_var = tk.StringVar(value=f"光标: ({cursor_x}, {cursor_y})")
    tk.Label(root, textvariable=coord_var, font=("Consolas", 10),
             bg="#0f3460", fg="#e94560", anchor="w", padx=5).pack(fill="x")

    tk_img_ref = [None]
    bg_image_id = canvas.create_image(0, 0, anchor="nw")
    latest_bg = [None]  # 后台线程写入的最新截图
    running = [True]

    def bg_capture():
        """后台线程：持续截屏。"""
        while running[0]:
            img = grab_screen()
            if img:
                latest_bg[0] = img.resize((cw, ch), Image.LANCZOS)
            time.sleep(0.15)

    t = threading.Thread(target=bg_capture, daemon=True)
    t.start()

    def refresh_screen():
        """主线程：合成光标并刷新画面。"""
        bg = latest_bg[0]
        if bg:
            img = bg.copy()
            draw = ImageDraw.Draw(img)
            cx = cursor_x // CANVAS_SCALE
            cy = cursor_y // CANVAS_SCALE
            r = 12
            draw.line([(cx - r, cy), (cx + r, cy)], fill="red", width=2)
            draw.line([(cx, cy - r), (cx, cy + r)], fill="red", width=2)
            draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)],
                         outline="red", width=2)
            tk_img_ref[0] = ImageTk.PhotoImage(img)
            canvas.itemconfig(bg_image_id, image=tk_img_ref[0])

        root.after(100, refresh_screen)

    refresh_screen()

    def on_key(event):
        nonlocal cursor_x, cursor_y
        k = event.keysym.lower()

        if k == 'i' or k == 'up':
            cursor_y = max(0, cursor_y - CURSOR_STEP)
        elif k == 'k' or k == 'down':
            cursor_y = min(SCREEN_H, cursor_y + CURSOR_STEP)
        elif k == 'j' or k == 'left':
            cursor_x = max(0, cursor_x - CURSOR_STEP)
        elif k == 'l' or k == 'right':
            cursor_x = min(SCREEN_W, cursor_x + CURSOR_STEP)
        elif k == 'w':
            send_action("Up")
            status_var.set("EOG: 上划")
        elif k == 's':
            send_action("Down")
            status_var.set("EOG: 下划")
        elif k == 'a':
            send_action("Left")
            status_var.set("EOG: 左划")
        elif k == 'd':
            send_action("Right")
            status_var.set("EOG: 右划")
        elif k == 'b' or k == 'space':
            tap_at(cursor_x, cursor_y)
            status_var.set(f"点击 ({cursor_x}, {cursor_y})")
        elif k == 'r':
            adb_key(4, "返回")
            status_var.set("返回上一页")
        elif k == 'h':
            adb_key(3, "主屏幕")
            status_var.set("回到主屏幕")
        elif k == 'q' or k == 'escape':
            running[0] = False
            root.destroy()
            return

        coord_var.set(f"光标: ({cursor_x}, {cursor_y})")

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", lambda: (running.__setitem__(0, False), root.destroy()))
    root.mainloop()


# ============================================================
# IMU 光标模式（模拟/串口）
# ============================================================
def imu_cursor_mode(port=None):
    """IMU 驱动光标 + 截屏镜像。无串口时用模拟数据。"""
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw
    import threading

    cw = SCREEN_W // CANVAS_SCALE
    ch = SCREEN_H // CANVAS_SCALE

    hmc = HeadMouseController(SCREEN_W, SCREEN_H)
    running = [True]
    latest_bg = [None]
    imu_status = ["等待IMU数据..."]

    # 尝试连接串口，失败则用模拟数据
    ser = None
    use_sim = True
    if port:
        try:
            import serial
            ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
            use_sim = False
            print(f"  IMU 串口已连接: {port}")
        except Exception as e:
            print(f"  串口连接失败: {e}，使用模拟数据")

    if use_sim:
        print("  使用模拟 IMU 数据（自动画圆）")

    root = tk.Tk()
    root.title("IMU 头控光标")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=cw, height=ch, bg="#1a1a2e")
    canvas.pack()

    status_var = tk.StringVar(value="空格=点击 / R=返回 / H=主屏 / C=校准 / Q=退出")
    tk.Label(root, textvariable=status_var, font=("Consolas", 10),
             bg="#16213e", fg="white", anchor="w", padx=5).pack(fill="x")

    coord_var = tk.StringVar(value="光标: (540, 960)")
    tk.Label(root, textvariable=coord_var, font=("Consolas", 10),
             bg="#0f3460", fg="#e94560", anchor="w", padx=5).pack(fill="x")

    tk_img_ref = [None]
    bg_image_id = canvas.create_image(0, 0, anchor="nw")

    def bg_capture():
        while running[0]:
            img = grab_screen()
            if img:
                latest_bg[0] = img.resize((cw, ch), Image.LANCZOS)
            time.sleep(0.15)

    threading.Thread(target=bg_capture, daemon=True).start()

    # IMU 数据读取线程
    sim_t = [0.0]

    def imu_read():
        while running[0]:
            if use_sim:
                # 模拟：光标画圆
                sim_t[0] += 0.05
                yaw = np.sin(sim_t[0]) * 8
                pitch = np.cos(sim_t[0]) * 6
                cx, cy = hmc.update(yaw, pitch)
                imu_status[0] = f"模拟 yaw={yaw:.1f} pitch={pitch:.1f}"
                time.sleep(0.05)
            else:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    try:
                        parts = line.split(',')
                        yaw = float(parts[0])
                        pitch = float(parts[1])
                        cx, cy = hmc.update(yaw, pitch)
                        imu_status[0] = f"yaw={yaw:.1f} pitch={pitch:.1f}"
                    except (ValueError, IndexError):
                        pass
                else:
                    time.sleep(0.01)

    threading.Thread(target=imu_read, daemon=True).start()

    def refresh_screen():
        bg = latest_bg[0]
        if bg:
            img = bg.copy()
            draw = ImageDraw.Draw(img)
            cx = int(hmc.cursor_x) // CANVAS_SCALE
            cy = int(hmc.cursor_y) // CANVAS_SCALE
            r = 12
            draw.line([(cx - r, cy), (cx + r, cy)], fill="red", width=2)
            draw.line([(cx, cy - r), (cx, cy + r)], fill="red", width=2)
            draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)],
                         outline="red", width=2)
            tk_img_ref[0] = ImageTk.PhotoImage(img)
            canvas.itemconfig(bg_image_id, image=tk_img_ref[0])
        coord_var.set(f"光标: ({int(hmc.cursor_x)}, {int(hmc.cursor_y)}) | {imu_status[0]}")
        root.after(80, refresh_screen)

    refresh_screen()

    def on_key(event):
        k = event.keysym.lower()
        if k == 'space' or k == 'b':
            tap_at(int(hmc.cursor_x), int(hmc.cursor_y))
            status_var.set(f"点击 ({int(hmc.cursor_x)}, {int(hmc.cursor_y)})")
        elif k == 'r':
            adb_key(4, "返回")
            status_var.set("返回上一页")
        elif k == 'h':
            adb_key(3, "主屏幕")
            status_var.set("回到主屏幕")
        elif k == 'c':
            hmc.is_initialized = False
            status_var.set("已校准，光标归中")
        elif k == 'q' or k == 'escape':
            running[0] = False
            if ser:
                ser.close()
            root.destroy()
            return

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", lambda: (
        running.__setitem__(0, False),
        ser.close() if ser else None,
        root.destroy()
    ))
    root.mainloop()


# ============================================================
# 键盘模拟模式
# ============================================================
KEY_MAP = {
    'a': 'Left',  'l': 'Left',
    'd': 'Right', 'r': 'Right',
    'w': 'Up',    'u': 'Up',
    's': 'Down',
    'b': 'Blink',
    ' ': 'Blink',  # 空格也触发点击
}

# Windows 方向键的特殊编码
ARROW_MAP = {
    b'K': 'Left',   # ←
    b'M': 'Right',  # →
    b'H': 'Up',     # ↑
    b'P': 'Down',   # ↓
}


def keyboard_mode():
    """键盘模拟模式：按键触发 ADB 命令。"""
    print("\n" + "=" * 50)
    print("键盘模拟模式")
    print("=" * 50)
    print("  WASD / 方向键 = 上下左右划")
    print("  B / 空格      = 点击（眨眼）")
    print("  Q / Esc        = 退出")
    print("=" * 50)

    if sys.platform == 'win32':
        import msvcrt
        while True:
            key = msvcrt.getch()
            if key == b'\xe0' or key == b'\x00':  # 方向键前缀
                arrow = msvcrt.getch()
                action = ARROW_MAP.get(arrow)
            elif key == b'\x1b' or key.lower() == b'q':  # Esc or Q
                print("\n  退出。")
                break
            else:
                action = KEY_MAP.get(key.decode('utf-8', errors='ignore').lower())

            if action:
                send_action(action)
    else:
        # Linux/Mac: 简单 input() 模式
        while True:
            try:
                ch = input("输入动作 (w/a/s/d/b, q退出): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if ch == 'q':
                break
            action = KEY_MAP.get(ch)
            if action:
                send_action(action)


# ============================================================
# 主函数
# ============================================================
def show_menu():
    """显示主菜单，返回用户选择。"""
    print("\n" + "=" * 50)
    print("  EOG + IMU -> ADB 屏幕控制")
    print("=" * 50)
    print("  1. 虚拟光标模式（截屏镜像 + IJKL/WASD）")
    print("  2. 键盘模拟模式（WASD 直接控制）")
    print("  3. 模拟信号测试（自动跑一轮）")
    print("  4. 串口实时模式（接 Arduino EOG）")
    print("  5. IMU 头控光标（模拟/串口）")
    print("  0. 退出")
    print("=" * 50)
    return input("  请选择: ").strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EOG -> ADB 屏幕控制")
    parser.add_argument('--width', type=int, default=1080, help='屏幕宽度')
    parser.add_argument('--height', type=int, default=1920, help='屏幕高度')
    parser.add_argument('--port', type=str, default=None, help='串口号 (如 COM3)')
    args = parser.parse_args()

    global SCREEN_W, SCREEN_H
    SCREEN_W = args.width
    SCREEN_H = args.height

    print("EOG -> ADB 屏幕控制 Demo")
    print(f"  屏幕分辨率: {SCREEN_W}x{SCREEN_H}")
    print(f"  冷却时间: {COOLDOWN_SEC}s")

    print("\n检查 ADB 连接...")
    check_adb()

    while True:
        choice = show_menu()
        if choice == '1':
            cursor_mode()
        elif choice == '2':
            keyboard_mode()
        elif choice == '3':
            sim_mode()
        elif choice == '4':
            live_mode(args.port)
        elif choice == '5':
            imu_cursor_mode(args.port)
        elif choice == '0':
            print("  再见。")
            break
        else:
            print("  无效选择，请重试。")


if __name__ == "__main__":
    main()
