# EOG+IMU Eye Control System

基于 EOG（眼电）和 IMU（惯性测量单元）的眼控鼠标系统，支持 PC 直接控制和 Android 虚拟机控制两种模式。

## 🎯 项目简介

通过 ESP32 边缘 AI 实时识别眼动和头部动作，支持：
- **PC 模式**：USB HID 直接控制电脑鼠标
- **Android 模式**：通过 ADB 控制 Android 虚拟机

## 🚀 核心特性

✅ 边缘 AI 部署 | ✅ 低延迟 <50ms | ✅ 准确率 >90% | ✅ 双模式支持

## 📁 Project Structure

```
├── DATA/
│   ├── EOG_data/                        # Raw EOG dataset
│   │   ├── 1/                           # Subject 1 (6 classes: Blink/Down/Left/Rest/Right/Up)
│   │   ├── 2/                           # Subject 2 (5 classes)
│   │   ├── 3/                           # Subject 3 (4 classes)
│   │   └── Test_Simulated/              # Simulated test data
│   ├── adb_control.py                   # ADB control utility
│   └── visualize_synthetic.py           # Dataset visualization
│
├── src/
│   ├── EOG_data_collection.py           # EOG signal acquisition script
│   └── generate_and_train_synthetic.py  # Synthetic data generation & training
│
├── final_submission/                    # Final deliverables
│   ├── IMU_EOG_Mouse.ino               # ESP32 firmware — PC USB HID mouse
│   ├── IMU_EOG_MuMu.ino                # ESP32 firmware — Android serial mode
│   ├── mumu_bridge.py                   # Python ADB bridge script
│   ├── EOG_AI_Engine_esp32_multi.h      # Pre-trained model (C++ header for ESP32)
│   └── Project_Report.pdf               # Project report
│
├── IMU_EOG/
│   ├── demo case/                       # Demo signal waveforms (6 classes)
│   └── output/                          # Captured EOG signal recordings
│
├── virtual/                             # Virtual/emulator deployment (pruned model)
│   ├── virtual.ino                      # ESP32 firmware for emulator
│   └── EOG_AI_Engine_esp32_multi_15tree.h
│
├── train.py                             # Random Forest model training script
├── EOG_data_collection.py               # EOG real-time data collection
├── realtime_visualization.py            # Real-time signal visualization
├── requirements.txt                     # Python dependencies
└── README.md
```

## 🔧 快速开始

**环境**：`pip install -r requirements.txt`

**PC模式**：烧录 `IMU_EOG_Mouse.ino` → 连接USB → 开始使用

**Android模式**：烧录 `IMU_EOG_MuMu.ino` → 运行 `python mumu_bridge.py --port COM3`

## 📊 动作映射

**PC**：上看→滚轮上 | 下看→滚轮下 | 左看→左键 | 右看→右键 | 头转→光标

**Android**：上看→上滑 | 下看→下滑 | 左看→点击/翻页 | 右看→返回 | 头转→虚拟光标

## 🎓 技术

Random Forest (15 trees) | 14维特征 | 推理<20ms | 采样50Hz