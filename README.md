# 双目视觉定位系统

## 项目简介

本项目是一个集成了深度学习算法的双目视觉系统，旨在实现对特定目标，如连接绳、铠装电缆、输油管道的实时检测、跟踪与三维空间定位，并利用 TensorRT 对深度估计和跟踪模型进行了加速。

## 主要功能

- **双目校正与深度估计**：支持传统的 StereoBM 算法及基于深度学习的 ACVNet 算法，生成高精度的深度图。
- **目标检测**：内置基于 YOLOv7 的检测模型，识别连接绳、电缆、管道等工业/水下目标。
- **实时跟踪**：利用 ViT 编码器和 SAM 解码器实现目标的稳定跟踪。
- **三维定位**：根据双目视差计算目标的 X, Y, Z 三维坐标。
- **数据传输**：通过 UDP 协议将定位信息实时发送至控制终端。
- **可视化界面**：提供基于 PyQt5 的图形用户界面，实时显示左右图像、深度图及检测定位结果。

## 核心技术栈

- **编程语言**：Python 3.x
- **计算机视觉**：OpenCV
- **深度学习框架**：PyTorch, TensorRT (用于模型加速)
- **GUI 框架**：PyQt5, Tkinter
- **检测模型**：YOLOv7
- **跟踪模型**：Mobile-SAM / ViT
- **深度估计**：Fast-ACVNet / StereoBM

## 目录结构说明

- `main.py`: 项目主程序入口，采用多线程处理图像采集、算法推理及数据发送。
- `configs/`: 配置文件目录，包含相机参数 `config.yaml` 及模型配置。
- `lib/`: 核心库目录
  - `depth_model.py`: 深度估计模型封装。
  - `detection_model.py`: 目标检测模型封装。
  - `tracking_model.py`: 目标跟踪模型封装。
  - `LocationModel.py`: 三维空间定位逻辑。
  - `Camera.py`: POE/RTSP 相机驱动封装。
  - `hk_camera.py`: 海康工业相机接口。
- `MvImport/`: 海康相机 SDK 适配层。
- `UDPtransmit/`: UDP 通信协议实现，用于数据外发。
- `stereo_models/`: 预训练模型文件存放地（.pt, .plan, .onnx）。
- `csrc/`: CUDA 加速相关的 C++ 源代码。

## 安装与配置

### 环境要求

- Windows 10/11 或 Linux

- CUDA 11.x + TensorRT 8.x (推荐，用于加速推理)

- Python 依赖库：

  ```bash
  pip install opencv-python numpy PyQt5 pyyaml easydict pillow
  ```

### 编译 CUDA 加速扩展

如果需要使用加速映射功能，请进入 `csrc` 目录并编译相关扩展：

```bash
cd csrc
python setup.py install
# 或者
python setup_map_cuda.py install
```

### 相机配置

编辑 `configs/config.yaml`，修改以下参数：

- `StereoCam`: 配置左右相机的 RTSP 地址或索引。
- `CameraRectify`: 填入相机的内参（K）和外参（R, T），以确保双目校正准确。
- `LocationUDP`: 设置接收定位数据的目标 IP 和端口。

## 使用方法

### 1. 启动定位系统

直接运行 `main.py` 启动后台推理与定位流程：

```bash
python main.py
```

### 2. 启动 GUI 界面

运行 `MainWindow.py` 或相关的界面脚本进行可视化监控：

```bash
python MainWindow_modify.py
```

### 3. 测试与调试

- `test.py`: 用于离线图片序列的检测与定位测试。
- `test-cam.py`: 用于单相机或双相机的连通性测试。

## 注意事项

- 确保 TensorRT 的 `.plan` 文件与当前的硬件环境（显卡驱动、TRT 版本）匹配。
- 双目定位精度极大程度上取决于 `config.yaml` 中相机标定参数的准确性。
- 改项目用于展示，部分代码和深度学习模型未提供
