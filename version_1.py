import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

import os


# nn.Linear(100, 256)里的100表示 输入二维张量的形状为 (batch_size, 100)，表示一个批次中有 batch_size 个样本，每个样本有100个特征。


# 定义生成器模型
class Generator(nn.Module): # 继承自nn.Module
    def __init__(self):  # self不能省略
        super(Generator, self).__init__() # 可以简写为 super().__init__()
        self.main = nn.Sequential(

            nn.Linear(100, 256),   # 输入输出都是二维张量，要求输入100个特征 输出256个特征
            nn.ReLU(True),   # ReLU激活函数，它将输入中的负值变为零，正值保持不变，默认是False

            nn.Linear(256, 512),  # 其中内置一个权重矩阵和一个偏置向量
            nn.ReLU(True),   # inplace=True：这个选项表示在原地修改输入张量，而不是创建一个新的张量。这可以节省内存

            nn.Linear(512, 1024),
            nn.ReLU(True),

            nn.Linear(1024, 28 * 28),
            nn.Tanh()    #  输入的数值映射到 (-1, 1) 区间
        )

    def forward(self, input):  #  正向传播
        return self.main(input)

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Linear(28 * 28, 1024),# 784
            nn.LeakyReLU(0.2, inplace=True), # 在输入值为负数时，输出值为输入值的0.2倍，而在输入值为正数时，输出保持不变

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()  # 输入的数值映射到 (0, 1)的范围内，Sigmoid和Tanh都不接受true参数
        )

    def forward(self, input):
        return self.main(input)

#  配置驱动

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
# 检查是否有可用的 MPS 设备
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("No CUDA or MPS device found, using CPU instead.")













# 创建生成器和判别器实例
generator = Generator().to(device)
discriminator = Discriminator().to(device)







# 定义数据加载器
#  transforms.ToTensor 将图像转换为 PyTorch 张量，并归一化为 [0, 1] 之间的值
#  transforms.Normalize((0.5,), (0.5,)) 表示将图像归一化为 [-1, 1]
#  综上将像素值从[0, 255]缩放到[0, 1]，然后再将它们标准化到[-1, 1]范围内。

# MNIST(""): 创建一个 MNIST 数据集的实例。括号内是数据的存储路径，这里使用了空字符串，表示将数据集下载到当前目录下。
# is_train: 表示是否加载训练集。is_train=True 时加载训练集，is_train=False 时加载测试集
# download=True: 表示如果数据集不存在，将自动下载数据集


# DataLoader对象 设置 shuffle=True，可以在每个epoch开始前打乱数据，
# data_loader = get_data_loader(is_train=True)
# for batch in data_loader:
# 处理每个批次的数据

# batch_size=64 表示每次加载的数据量，即批量大小为 64 个样本。



def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # to_tensor相当于一个组合转换器
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=64, shuffle=True)






# 定义损失函数和优化器
#  nn.BCELoss(output, real_labels)  这种损失函数通常用于二分类任务中，评估模型预测的概率与真实标签之间的差异，值越小越精准。
#  使用 Adam 优化器来更新判别器的参数，学习率（lr）设为 0.0002。
#  discriminator.parameters()：
#  这是一个方法调用，它返回判别器模型中所有需要优化的参数。通过传递这些参数，优化器可以对模型的权重和偏置进行更新



criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002)
optimizerG = optim.Adam(generator.parameters(), lr=0.0002)








# 训练过程

# 元组，列表都可以嵌套
#  (data,)明确这是一个包含一个元素的元组，去掉则不是元组类型
# (data, _): 包含两个元素的元组。data 代表元组中的第一个元素。_代表元组中的第二个元素（该代码中表示忽略标签）并且被忽略。
# dataloader每个元组里必须要有两个元素，具体是 (data, labels) 的元组
# enumerate(dataloader)表示为每个元组生成一个（索引, 元组）元组，i取索引，(data, _)取元组中的第一个元素
# 使用enumerate(dataloader)是为了提供索引编号，使得在遍历数据时可以同时获取当前批次的索引和数据


# data 的原始形状 (batch_size, 1, 28, 28) 1：表示通道数，对于灰度图像，通道数为1。如果是彩色图像，通道数通常是3
# 28：高度，28：宽度
# view 函数被用来将二维图像展平为一维向量，以便输入到全连接层中
# data.view(-1, 28*28) -1 代表当前维度大小根据data 的第一个维度（即批次大小 batch_size）自动调整。
#  28*28 是因为输入数据是 28x28 像素的图像，将其展平为一维张量即二维（batch_size，28x28）里的28x28，以便输入到线性层
# 原始形状: torch.Size([64, 1, 28, 28])
# 展平后的形状: torch.Size([64, 784])



# EG 创建一个形状为 [64, 3, 28, 28]  [batch_size, channels, height, width]的张量
# data = torch.randn(64, 3, 28, 28)
# 使用 .view(-1, 28*28) 将张量重塑为 [192, 784]  将每张 28x28 的二维图像展平成一个一维向量，每个向量有 784 个元素，有192个批次
# flattened_data = data.view(-1, 28*28)

# torch.full((3, 2), 7)
# tensor([[7, 7],
#        [7, 7],
#        [7, 7]])











def train_gan(generator, discriminator, dataloader, num_epochs=10, start_epoch=0):
    for epoch in range(start_epoch, start_epoch+num_epochs):  # 生成0到49的整数序列，迭代50次
        for i, (data, _) in enumerate(dataloader):



            # 更新判别器
            optimizerD.zero_grad() # 作用与discriminator.zero_grad() 一样：梯度清零

            real_data = data.view(-1, 28*28).to(device)
            batch_size = real_data.size(0) # 获取批次大小，取第一个元素的位置
            real_labels = torch.full((batch_size, 1), 1, device=device, dtype=torch.float)
            fake_labels = torch.full((batch_size, 1), 0, device=device, dtype=torch.float)





            # 训练判别器: 用真实数据


            output = discriminator(real_data)
            lossD_real = criterion(output, real_labels)
            lossD_real.backward()
            # 通过反向传播算法，计算出损失函数在当前参数值处的变化率（梯度：指向损失增加最快的方向。为了最小化损失，需要沿着梯度的反方向调整参数）


            optimizerD.step() # 调用优化器Adam(已事先声明)，使用计算出的梯度更新模型参数




            # 训练判别器: 用假数据
            # torch.randn 生成一个服从标准正态分布（均值为0，标准差为1）的张量
            # batch_size 张量的第一个维度，决定了一次传递给模型的样本数量
            # 100 张量的第二个维度，它表示输入元素数量。

            # detach() 对假数据使用，使其不会影响之前计算图中的梯度,确保生成器和判别器的梯度更新过程是分离的
            optimizerD.zero_grad()

            noise = torch.randn(batch_size, 100, device=device)
            fake_data = generator(noise)
            output = discriminator(fake_data.detach())
            # 在训练判别器时，你确实需要使用 detach() 来确保生成器的梯度不会传递到判别器中。

            lossD_fake = criterion(output, fake_labels)
            lossD_fake.backward()

            optimizerD.step()











            # 更新生成器
           # generator.zero_grad()
            optimizerG.zero_grad()

            output = discriminator(fake_data)
        # 不需要detach()，因为如果你在这里使用 detach()，则生成器的输出与判别器的输入之间将没有梯度连接，生成器的参数将无法更新
        # 总结 生成器的梯度不需要传递到判别器中，判别器的梯度需要传递到生成器中
            lossG = criterion(output, real_labels)
        #  生成器通过假数据被贴上真标签来迷惑判别器
        #  生成器的损失函数是基于判别器的输出计算的，即判别器认为生成数据是真数据的概率。
        #  通过将生成数据贴上真标签，生成器可以通过梯度下降来优化生成的数据，使其更像真实数据


            lossG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{start_epoch+num_epochs}] Batch {i}/{len(dataloader)} '
                      f'Loss D: {lossD_real + lossD_fake:.4f}, Loss G: {lossG:.4f}')
# 在训练的初期，Loss D 会快速下降,Loss G 往往较高
# 因为鉴别器能够轻松地区分真实数据和生成数据,生成器生成的假数据很容易被鉴别器识别

        # 保存生成器和判别器的状态
        save_checkpoint(generator, discriminator, optimizerG, optimizerD, epoch)




########### 保存模型


def save_checkpoint(generator, discriminator, optimizerG, optimizerD, epoch):
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'epoch': epoch,
    }
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(checkpoint, f'checkpoints/gan_epoch_{epoch}.pth')

def load_checkpoint(generator, discriminator, optimizerG, optimizerD, file_path):
    checkpoint = torch.load(file_path, weights_only=True)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    return checkpoint['epoch']

# 恢复训练，例如从 epoch 10 开始继续训练
checkpoint_path = 'checkpoints/gan_epoch_12.pth'
if os.path.exists(checkpoint_path):
    start_epoch =+ load_checkpoint(generator, discriminator, optimizerG, optimizerD, checkpoint_path)
else:
    start_epoch = 0





train_data = get_data_loader(is_train=True)
train_gan(generator, discriminator, train_data, num_epochs=9,start_epoch=start_epoch)


# 保存生成器
# torch.save(generator.state_dict(), 'generator.pth')

# 保存鉴别器
# torch.save(discriminator.state_dict(), 'discriminator.pth')



# 生成样本并可视化
def generate_and_plot_samples(generator, num_samples=16):
    generator.eval()  # 切换到评估模式 确保生成样本时模型的行为稳定
    noise = torch.randn(num_samples, 100, device=device)
    with torch.no_grad():
        generated_data = generator(noise).cpu().view(num_samples, 28, 28)
    for i in range(num_samples):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_data[i], cmap='gray')
        plt.axis('off')
    plt.show()

    generator.train()  # 切换回训练模式 以便继续训练
generate_and_plot_samples(generator)
