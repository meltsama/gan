
# GAN 手写数字生成器

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于 PyTorch 实现的生成对抗网络（GAN），用于生成 MNIST 手写数字图像。

## 项目特点
- **完整GAN实现**：包含生成器（Generator）和判别器（Discriminator）网络
- **断点续训功能**：支持保存/加载模型检查点（checkpoint）
- **多设备支持**：自动检测并优先使用 CUDA/MPS 加速
- **实时可视化**：训练过程中可查看生成样本的演变
- **模块化设计**：代码结构清晰，易于二次开发

## 环境要求
- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy

## 快速开始

### 安装依赖
```
bash
pip install torch torchvision matplotlib numpy
```
### 训练模型
```
bash
默认训练50个epoch
python main.py

# 从检查点恢复训练（示例从epoch10继续）
python main.py --resume checkpoints/gan_epoch_10.pth

# 自定义训练轮数
python main.py --epochs 100
```
### 生成样本
```
python
from main import generate_and_plot_samples

# 生成16个样本并显示
generate_and_plot_samples(generator)
```
## 核心实现

### 网络架构

#### 生成器（Generator）
```
python
Sequential(
  Linear(100 → 256), ReLU(),
  Linear(256 → 512), ReLU(),
  Linear(512 → 1024), ReLU(),
  Linear(1024 → 784), Tanh()  # 输出归一化到[-1,1]
)
```
#### 判别器（Discriminator）
```
python
Sequential(
  Linear(784 → 1024), LeakyReLU(0.2),
  Linear(1024 → 512), LeakyReLU(0.2),
  Linear(512 → 256), LeakyReLU(0.2),
  Linear(256 → 1), Sigmoid()  # 输出[0,1]概率值
)
```
### 关键参数
| 参数               | 值       | 说明                       |
|--------------------|----------|---------------------------|
| 潜在维度           | 100      | 生成器输入噪声的维度        |
| 学习率             | 0.0002   | Adam优化器的初始学习率      |
| 批量大小           | 64       | 每次训练使用的样本数        |
| 默认训练轮数       | 50       | 完整训练流程的迭代次数      |

## 训练效果
不同训练阶段的生成样本对比：

| Epoch 30          | Epoch 40          | Epoch 50          |
|-------------------|-------------------|-------------------|
| ![E10](assets/epoch10.png) | ![E30](https://github.com/meltsama/gan/blob/fe42bfb4c6b9cd4bb02858d23d5706341df52f03/Epoch_40.png) | ![E50](https://github.com/meltsama/gan/blob/e5ff58d2979206819be977ff25af6e994a3232ac/Epoch_50.png) |

## 文件结构
```

.
├── main.py             # 主程序入口
├── checkpoints/        # 模型检查点保存目录
├── utils.py            # 工具函数（生成样本/加载数据等）
├── README.md           # 项目说明文档
└── requirements.txt    # 依赖列表
```
## 常见问题

### 为什么生成的图像模糊？
- 增加训练轮数（建议至少50个epoch）
- 尝试调整学习率（0.0001~0.0005）
- 检查网络结构是否合理

### 如何提高训练速度？
- 使用CUDA显卡加速
- 适当增大批量大小（batch_size）
- 减少全连接层的神经元数量

### 恢复训练时遇到错误？
- 确保检查点文件路径正确
- 检查PyTorch版本是否兼容
- 确认网络结构没有改动

## 许可协议
本项目采用 MIT 许可证 - 详情参见 [LICENSE](LICENSE) 文件。


 

