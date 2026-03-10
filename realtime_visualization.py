"""
Arduino传感器数据实时可视化程序
功能：实时接收串口数据并绘制图表

使用前确保安装：
pip install pyserial matplotlib numpy pandas
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime

# ========== 配置 ==========
SERIAL_PORT = 'COM4'  # Windows: COM3, Mac: /dev/tty.usbmodem*, Linux: /dev/ttyACM0
BAUD_RATE = 9600
MAX_POINTS = 300  
SAVE_TO_FILE = True  

# ========== 数据存储 ==========
data_buffer = {
    'time': deque(maxlen=MAX_POINTS),
    'force': deque(maxlen=MAX_POINTS),
    'flex': deque(maxlen=MAX_POINTS),
    'ax': deque(maxlen=MAX_POINTS),
    'ay': deque(maxlen=MAX_POINTS),
    'az': deque(maxlen=MAX_POINTS),
    'gx': deque(maxlen=MAX_POINTS),
    'gy': deque(maxlen=MAX_POINTS),
    'gz': deque(maxlen=MAX_POINTS),
    'mx': deque(maxlen=MAX_POINTS),
    'my': deque(maxlen=MAX_POINTS),
    'mz': deque(maxlen=MAX_POINTS)
}

# 用于保存完整数据
all_data = []
is_recording = False

# ========== 连接Arduino ==========
print("="*50)
print("Arduino传感器数据可视化系统")
print("="*50)
print(f"\n正在连接 {SERIAL_PORT}...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"✅ 已连接到 {SERIAL_PORT}")
    print("\n等待Arduino启动...")
    import time
    time.sleep(2)  # 等待Arduino复位
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("\n请检查：")
    print("1. Arduino是否已连接")
    print("2. 串口号是否正确")
    print("3. Arduino IDE的串口监视器是否已关闭")
    exit()

# ========== 创建图表 ==========
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 10))
fig.canvas.manager.set_window_title('Arduino多传感器实时监测')
fig.suptitle('🎯 多传感器数据实时可视化系统', fontsize=16, fontweight='bold')

# 子图1: Force 和 Flex
ax1 = plt.subplot(3, 3, 1)
ax1.set_title('Force & Flex Sensors', fontsize=10, fontweight='bold')
line_force, = ax1.plot([], [], 'r-', label='Force', linewidth=2)
ax1.set_ylabel('Force', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim(-50, 1050)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
line_flex, = ax1_twin.plot([], [], 'b-', label='Flex', linewidth=2)
ax1_twin.set_ylabel('Flex', color='blue')
ax1_twin.tick_params(axis='y', labelcolor='blue')
ax1_twin.set_ylim(-100, 300)
ax1_twin.legend(loc='upper right')

# 子图2-4: 加速度 XYZ
ax2 = plt.subplot(3, 3, 2)
ax2.set_title('Acceleration X', fontsize=10, fontweight='bold')
line_ax, = ax2.plot([], [], 'g-', linewidth=2)
ax2.set_ylabel('Accel X (g)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylim(-2, 2)
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 3, 3)
ax3.set_title('Acceleration Y', fontsize=10, fontweight='bold')
line_ay, = ax3.plot([], [], 'm-', linewidth=2)
ax3.set_ylabel('Accel Y (g)')
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.set_ylim(-2, 2)
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 3, 4)
ax4.set_title('Acceleration Z', fontsize=10, fontweight='bold')
line_az, = ax4.plot([], [], 'c-', linewidth=2)
ax4.set_ylabel('Accel Z (g)')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylim(-2, 2)
ax4.grid(True, alpha=0.3)

# 子图5: 陀螺仪 XYZ
ax5 = plt.subplot(3, 3, 5)
ax5.set_title('Gyroscope (XYZ)', fontsize=10, fontweight='bold')
line_gx, = ax5.plot([], [], 'r-', label='Gyro X', linewidth=1.5, alpha=0.8)
line_gy, = ax5.plot([], [], 'g-', label='Gyro Y', linewidth=1.5, alpha=0.8)
line_gz, = ax5.plot([], [], 'b-', label='Gyro Z', linewidth=1.5, alpha=0.8)
ax5.set_ylabel('Angular Velocity')
ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax5.set_ylim(-5000, 5000)
ax5.legend(loc='upper right', fontsize=8)
ax5.grid(True, alpha=0.3)

# 子图6: 磁力计 XYZ
ax6 = plt.subplot(3, 3, 6)
ax6.set_title('Magnetometer (XYZ)', fontsize=10, fontweight='bold')
line_mx, = ax6.plot([], [], 'r-', label='Mag X', linewidth=1.5, alpha=0.8)
line_my, = ax6.plot([], [], 'g-', label='Mag Y', linewidth=1.5, alpha=0.8)
line_mz, = ax6.plot([], [], 'b-', label='Mag Z', linewidth=1.5, alpha=0.8)
ax6.set_ylabel('Magnetic Field')
ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax6.legend(loc='upper right', fontsize=8)
ax6.grid(True, alpha=0.3)

# 子图7: 3D加速度轨迹
ax7 = plt.subplot(3, 3, 7, projection='3d')
ax7.set_title('3D Acceleration', fontsize=10, fontweight='bold')
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.set_zlabel('Z')
line_3d, = ax7.plot([], [], [], 'b-', linewidth=1, alpha=0.6)

# 子图8: 统计信息
ax8 = plt.subplot(3, 3, 8)
ax8.set_title('Statistics', fontsize=10, fontweight='bold')
ax8.axis('off')
stats_text = ax8.text(0.05, 0.95, '', fontsize=9, family='monospace',
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 子图9: 系统状态
ax9 = plt.subplot(3, 3, 9)
ax9.set_title('System Status', fontsize=10, fontweight='bold')
ax9.axis('off')
status_text = ax9.text(0.05, 0.95, '', fontsize=10, family='monospace',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()

# ========== 数据解析 ==========
def parse_data(line):
    """解析Arduino发送的CSV数据"""
    try:
        # 数据格式: time,force,flex,ax,ay,az,gx,gy,gz,mx,my,mz
        parts = line.strip().split(',')
        if len(parts) == 12:
            data = {
                'time': float(parts[0]),
                'force': int(parts[1]),
                'flex': int(parts[2]),
                'ax': float(parts[3]),
                'ay': float(parts[4]),
                'az': float(parts[5]),
                'gx': int(parts[6]),
                'gy': int(parts[7]),
                'gz': int(parts[8]),
                'mx': int(parts[9]),
                'my': int(parts[10]),
                'mz': int(parts[11])
            }
            return data
    except:
        pass
    return None

# ========== 更新统计信息 ==========
def update_stats():
    """计算并显示统计信息"""
    if len(data_buffer['force']) < 2:
        return "等待数据..."
    
    stats = f"""
╔══════════════════════════╗
║  数据统计 ({len(data_buffer['time'])} 点)
╠══════════════════════════╣
║ Force:
║   均值: {np.mean(data_buffer['force']):.1f}
║   最大: {np.max(data_buffer['force']):.0f}
║   最小: {np.min(data_buffer['force']):.0f}
║
║ Flex:
║   均值: {np.mean(data_buffer['flex']):.1f}
║   最大: {np.max(data_buffer['flex']):.0f}
║   最小: {np.min(data_buffer['flex']):.0f}
║
║ Accel (g):
║   X: {np.mean(data_buffer['ax']):.2f}
║   Y: {np.mean(data_buffer['ay']):.2f}
║   Z: {np.mean(data_buffer['az']):.2f}
╚══════════════════════════╝
"""
    return stats

# ========== 更新状态信息 ==========
def update_status():
    """显示系统状态"""
    status = f"""
{'🔴 录制中' if is_recording else '⚪ 待机中'}

总数据点: {len(all_data)}

操作提示：
• Arduino按钮控制录制
• 关闭窗口保存数据
"""
    return status

# ========== 动画更新函数 ==========
def animate(frame):
    """每帧更新数据和图表"""
    global is_recording
    
    # 读取串口数据
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # 检测录制状态
            if 'recording' in line.lower():
                is_recording = True
                print("\n🔴 开始录制")
                continue
            elif 'stop recording' in line.lower():
                is_recording = False
                print("⚫ 停止录制")
                continue
            
            # 解析数据
            data = parse_data(line)
            if data:
                # 添加到缓冲区
                for key, value in data.items():
                    data_buffer[key].append(value)
                
                # 保存到完整数据
                if is_recording:
                    all_data.append(data)
                
        except Exception as e:
            pass
    
    # 更新图表
    if len(data_buffer['time']) > 1:
        # 转换为相对时间
        times = list(data_buffer['time'])
        t_rel = [(t - times[0]) / 1000.0 for t in times]  # 转为秒
        
        # 更新各条曲线
        line_force.set_data(t_rel, list(data_buffer['force']))
        line_flex.set_data(t_rel, list(data_buffer['flex']))
        line_ax.set_data(t_rel, list(data_buffer['ax']))
        line_ay.set_data(t_rel, list(data_buffer['ay']))
        line_az.set_data(t_rel, list(data_buffer['az']))
        line_gx.set_data(t_rel, list(data_buffer['gx']))
        line_gy.set_data(t_rel, list(data_buffer['gy']))
        line_gz.set_data(t_rel, list(data_buffer['gz']))
        line_mx.set_data(t_rel, list(data_buffer['mx']))
        line_my.set_data(t_rel, list(data_buffer['my']))
        line_mz.set_data(t_rel, list(data_buffer['mz']))
        
        # 更新3D轨迹
        ax_list = list(data_buffer['ax'])[-50:]  # 最近50个点
        ay_list = list(data_buffer['ay'])[-50:]
        az_list = list(data_buffer['az'])[-50:]
        line_3d.set_data(ax_list, ay_list)
        line_3d.set_3d_properties(az_list)
        
        # 自动调整X轴范围
        x_min = max(0, t_rel[-1] - 10)
        x_max = t_rel[-1] + 1
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_xlim(x_min, x_max)
        
        # 自动调整磁力计Y轴
        if len(data_buffer['mx']) > 0:
            mx_data = list(data_buffer['mx'])
            my_data = list(data_buffer['my'])
            mz_data = list(data_buffer['mz'])
            mag_min = min(min(mx_data), min(my_data), min(mz_data))
            mag_max = max(max(mx_data), max(my_data), max(mz_data))
            ax6.set_ylim(mag_min - 100, mag_max + 100)
        
        # 更新统计和状态
        stats_text.set_text(update_stats())
        status_text.set_text(update_status())

# ========== 启动动画 ==========
print("\n" + "="*50)
print("📈 可视化系统已启动！")
print("="*50)
print("\n操作说明：")
print("1. 按Arduino上的按钮开始录制")
print("2. 操作传感器，观察图表变化")
print("3. 再次按按钮停止录制")
print("4. 关闭窗口保存数据并退出")
print("="*50 + "\n")

ani = animation.FuncAnimation(fig, animate, interval=50, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("\n程序中断")

# ========== 保存数据 ==========
if SAVE_TO_FILE and len(all_data) > 0:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sensor_data_{timestamp}.csv"
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    
    print(f"\n✅ 数据已保存到: {filename}")
    print(f"   共 {len(all_data)} 条数据")
else:
    print("\n⚠️  没有录制数据，未保存文件")

# 关闭串口
ser.close()
print("串口已关闭")