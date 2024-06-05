import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
from einops import rearrange, repeat
from torch.utils.data import DataLoader
import os
import time
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset

# 定义一个自定义的数据集类，应用 Mixup 和 CutMix 数据增强
class CustomCIFAR10(Dataset):
    def __init__(self, dataset, mixup_alpha=0.2, cutmix_alpha=0.2, prob=0.5):
        self.dataset = dataset  # 原始数据集
        self.mixup_alpha = mixup_alpha  # Mixup 参数 alpha
        self.cutmix_alpha = cutmix_alpha  # CutMix 参数 alpha
        self.prob = prob  # 应用 Mixup 或 CutMix 的概率
        self.num_classes = 10  # CIFAR-10 有10个类别

    def __len__(self):
        return len(self.dataset)  # 返回数据集的长度

    def __getitem__(self, index):
        img, target = self.dataset[index]  # 获取图像和标签
        target = self.to_one_hot(target)  # 将标签转换为 one-hot 编码

        if torch.rand(1).item() < self.prob:  # 按照设定的概率应用 Mixup 或 CutMix
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item() if torch.rand(1).item() < 0.5 else torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            rand_index = torch.randint(0, len(self.dataset), (1,)).item()  # 随机选择另一个样本
            img2, target2 = self.dataset[rand_index]
            target2 = self.to_one_hot(target2)

            if lam < 0.5:  # 应用 Mixup
                img = lam * img + (1 - lam) * img2
                target = lam * target + (1 - lam) * target2
            else:  # 应用 CutMix
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lam)
                img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
                target = lam * target + (1 - lam) * target2

        return img, target

    def rand_bbox(self, size, lam):
        W = size[1]  # 图像的宽度
        H = size[2]  # 图像的高度
        cut_rat = torch.sqrt(torch.tensor(1. - lam))  # 计算裁剪比例
        cut_w = torch.round(W * cut_rat).int()  # 计算裁剪区域的宽度
        cut_h = torch.round(H * cut_rat).int()  # 计算裁剪区域的高度

        cx = torch.randint(W, (1,)).item()  # 随机选择裁剪区域的中心 x 坐标
        cy = torch.randint(H, (1,)).item()  # 随机选择裁剪区域的中心 y 坐标

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W).item()  # 计算裁剪区域的左上角 x 坐标
        bby1 = torch.clamp(cy - cut_h // 2, 0, H).item()  # 计算裁剪区域的左上角 y 坐标
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W).item()  # 计算裁剪区域的右下角 x 坐标
        bby2 = torch.clamp(cy + cut_h // 2, 0, H).item()  # 计算裁剪区域的右下角 y 坐标

        return bbx1, bby1, bbx2, bby2

    def to_one_hot(self, target):
        one_hot = torch.zeros(self.num_classes, dtype=torch.float)  # 创建一个零向量
        one_hot[target] = 1.0  # 将目标位置设置为1
        return one_hot

# Attention 结构
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 计算内部维度
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要输出投影

        self.heads = heads  # 头的数量
        self.scale = dim_head ** -0.5  # 缩放因子

        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.attend = nn.Softmax(dim=-1)  # Softmax 注意力
        self.dropout = nn.Dropout(dropout)  # Dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 线性层，用于生成查询、键和值

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 输出线性层
            nn.Dropout(dropout)  # Dropout
        ) if project_out else nn.Identity()  # 如果需要投影，则使用线性层和 Dropout，否则使用 Identity

    def forward(self, x):
        x = self.norm(x)  # 层归一化
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 生成查询、键和值，并沿最后一个维度分块
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # 重排列为多头注意力格式

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算注意力得分
        attn = self.attend(dots)  # 应用 Softmax
        attn = self.dropout(attn)  # 应用 Dropout

        out = torch.matmul(attn, v)  # 计算注意力输出
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排列为原格式
        return self.to_out(out)  # 返回输出

# ViT 整体结构
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = image_size, image_size  # 图像尺寸
        patch_height, patch_width = patch_size, patch_size  # Patch 尺寸
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'  # 确保图像尺寸能被 patch 尺寸整除

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 计算 patch 数量
        patch_dim = channels * patch_height * patch_width  # 计算每个 patch 的维度

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),  # 层归一化
            nn.Linear(patch_dim, dim),  # 线性层
            nn.LayerNorm(dim),  # 层归一化
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类 token
        self.dropout = nn.Dropout(emb_dropout)  # Dropout

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer 编码器

        self.pool = pool  # 池化方式
        self.to_latent = nn.Identity()  # 恒等层
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_classes)  # 线性层，输出类别概率
        )

    def forward(self, img):
        p1, p2 = img.shape[-2] // patch_size, img.shape[-1] // patch_size  # 计算每个方向的 patch 数量
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)  # 将图像分块并展平
        x = self.to_patch_embedding(x)  # 计算 patch 嵌入
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # 生成分类 token
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接分类 token 和 patch 嵌入
        x += self.pos_embedding[:, :(n + 1)]  # 添加位置嵌入
        x = self.dropout(x)  # 应用 Dropout

        x = self.transformer(x)  # 通过 Transformer 编码器

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # 池化
        x = self.to_latent(x)  # 恒等层
        return self.mlp_head(x)  # 分类头

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])  # 多层 Transformer
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # 注意力层
                FeedForward(dim, mlp_dim, dropout=dropout)  # 前馈层
            ]))
        self.norm = nn.LayerNorm(dim)  # 层归一化

    def forward(self, x):
        for attn, ff in self.layers:  # 逐层计算
            x = attn(x) + x  # 残差连接
            x = ff(x) + x  # 残差连接
        return self.norm(x)  # 层归一化

# 前向 MLP 网络
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, hidden_dim),  # 线性层
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout
            nn.Linear(hidden_dim, dim),  # 线性层
            nn.Dropout(dropout)  # Dropout
        )

    def forward(self, x):
        return self.net(x)  # 前向计算

# 平滑交叉熵损失函数
class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing  # 平滑参数

    def forward(self, preds, targets):
        log_preds = torch.nn.functional.log_softmax(preds, dim=-1)  # 计算 log_softmax
        nll_loss = -log_preds * targets  # 计算负对数似然损失
        nll_loss = nll_loss.sum(-1)  # 按类别求和
        return nll_loss.mean()  # 求均值

# 训练函数
def train(epoch, net, trainloader, criterion, optimizer, scaler, accumulation_steps):
    print(f'\nEpoch: {epoch}')  # 打印当前 epoch
    net.train()  # 设置模型为训练模式
    train_loss = 0  # 初始化训练损失
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总样本数

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start_time = time.time()  # 记录起始时间

        inputs, targets = inputs.to(device), targets.to(device)  # 将输入和标签移动到 GPU
        optimizer.zero_grad()  # 梯度清零

        with autocast():  # 自动混合精度
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, targets) / accumulation_steps  # 计算损失并除以累积步数

        scaler.scale(loss).backward()  # 反向传播并缩放损失

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器
            optimizer.zero_grad()  # 梯度清零

        end_time = time.time()  # 记录结束时间
        batch_time = end_time - start_time  # 计算批次时间

        train_loss += loss.item() * accumulation_steps  # 累加训练损失
        _, predicted = outputs.max(1)  # 获取预测值
        total += targets.size(0)  # 累加样本数
        correct += predicted.eq(targets.max(1)[1]).sum().item()  # 累加正确预测数

        print(f'Batch {batch_idx + 1}/{len(trainloader)} | Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total}) | Time: {batch_time:.3f}s')

    return train_loss / (batch_idx + 1)  # 返回平均训练损失

# 测试函数
def test(epoch, net, testloader, criterion, scheduler, best_acc):
    net.eval()  # 设置模型为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总样本数

    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (inputs, targets) in enumerate(testloader):
            start_time = time.time()  # 记录起始时间

            inputs, targets = inputs.to(device), targets.to(device)  # 将输入和标签移动到 GPU
            targets = to_one_hot(targets, num_classes=10)  # 转换为 one-hot 编码
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失

            end_time = time.time()  # 记录结束时间
            batch_time = end_time - start_time  # 计算批次时间

            test_loss += loss.item()  # 累加测试损失
            _, predicted = outputs.max(1)  # 获取预测值
            total += targets.size(0)  # 累加样本数
            correct += predicted.eq(targets.max(1)[1]).sum().item()  # 累加正确预测数

            print(f'Batch {batch_idx + 1}/{len(testloader)} | Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total}) | Time: {batch_time:.3f}s')

    acc = 100. * correct / total  # 计算准确率
    if acc > best_acc:  # 如果当前准确率大于最佳准确率
        print('Saving...')  # 打印保存信息
        state = {
            'net': net.state_dict(),  # 保存模型参数
            'acc': acc,  # 保存当前准确率
            'epoch': epoch,  # 保存当前 epoch
        }
        if not os.path.isdir('checkpoint'):  # 如果 checkpoint 文件夹不存在
            os.mkdir('checkpoint')  # 创建 checkpoint 文件夹
        torch.save(state, f'./checkpoint/vit_cifar10_ckpt.pth')  # 保存模型
        best_acc = acc  # 更新最佳准确率

    scheduler.step()  # 更新学习率
    return test_loss / (batch_idx + 1), acc, best_acc  # 返回平均测试损失、当前准确率和最佳准确率

# 将标签转换为 one-hot 编码
def to_one_hot(target, num_classes):
    one_hot = torch.zeros((target.size(0), num_classes), device=target.device)  # 创建一个全零矩阵
    one_hot.scatter_(1, target.unsqueeze(1), 1.0)  # 在目标位置设置为1
    return one_hot

if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有 GPU，则使用 GPU，否则使用 CPU
    if device == 'cpu':
        print("No GPU found. Using CPU.")  # 如果没有 GPU，则打印提示信息

    # 加载和预处理数据集
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整到 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        RandAugment(),  # 添加 RandAugment 数据增强
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    trans_valid = transforms.Compose([
        transforms.Resize(256),  # 调整到 256x256
        transforms.CenterCrop(224),  # 中心裁剪到 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=trans_train)  # 加载训练集并应用变换
    trainset = CustomCIFAR10(trainset)  # 使用 CustomCIFAR10 进行数据增强

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)  # 创建 DataLoader

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=trans_valid)  # 加载测试集并应用变换
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)  # 创建 DataLoader

    # 初始化模型、损失函数和优化器
    patch_size = 16  # Patch 大小

    net = ViT(
        image_size=224,  # 图像大小
        patch_size=patch_size,  # Patch 大小
        num_classes=10,  # 类别数
        dim=512,  # 模型维度
        depth=6,  # Transformer 层数
        heads=8,  # 注意力头数
        mlp_dim=2048,  # MLP 维度
        dropout=0.1,  # Dropout 率
        emb_dropout=0.1  # Embedding Dropout 率
    ).to(device)  # 将模型移动到 GPU

    # 使用 DataParallel 进行多GPU训练
    net = nn.DataParallel(net)

    criterion = SmoothCrossEntropyLoss(smoothing=0.1)  # 使用平滑交叉熵损失函数
    optimizer = optim.AdamW(net.parameters(), lr=3e-4)  # 使用 AdamW 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # 学习率调度器
    scaler = GradScaler()  # 混合精度缩放器

    best_acc = 0  # 初始化最佳准确率
    accumulation_steps = 2  # 梯度累积步数

    # 训练和测试模型
    for epoch in range(1, 201):  # 训练 200 个 epoch
        epoch_start_time = time.time()  # 记录 epoch 开始时间

        train_loss = train(epoch, net, trainloader, criterion, optimizer, scaler, accumulation_steps)  # 训练模型
        test_loss, test_acc, best_acc = test(epoch, net, testloader, criterion, scheduler, best_acc)  # 测试模型

        epoch_end_time = time.time()  # 记录 epoch 结束时间
        epoch_time = epoch_end_time - epoch_start_time  # 计算 epoch 时间

        # 清理显存
        torch.cuda.empty_cache()

        # 记录日志
        content = f'{epoch} | lr: {optimizer.param_groups[0]["lr"]:.7f} | train loss: {train_loss:.3f} | test loss: {test_loss:.3f} | test acc: {test_acc:.3f}% | epoch time: {epoch_time:.3f}s\n'
        with open('log_vit_cifar10.txt', 'a') as f:
            f.write(content)  # 写入日志文件

        print(f'Epoch {epoch} completed in {epoch_time:.3f} seconds')  # 打印 epoch 完成信息
