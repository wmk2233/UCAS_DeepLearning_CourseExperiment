import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import os

# ViT 模型定义中的注意力机制模块
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 内部维度 = 头的数量 * 每个头的维度
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要输出投影

        self.heads = heads  # 头的数量
        self.scale = dim_head ** -0.5  # 缩放因子

        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.attend = nn.Softmax(dim=-1)  # 在最后一个维度上应用 Softmax
        self.dropout = nn.Dropout(dropout)  # Dropout 层

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 线性层，用于生成查询、键和值

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 输出线性层
            nn.Dropout(dropout)  # Dropout 层
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

# ViT 模型定义
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
        self.dropout = nn.Dropout(emb_dropout)  # Dropout 层

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

# Transformer 结构定义
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])  # 创建一个模块列表来存储多层 Transformer
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
            nn.Dropout(dropout),  # Dropout 层
            nn.Linear(hidden_dim, dim),  # 线性层
            nn.Dropout(dropout)  # Dropout 层
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

# 准备数据集和数据加载器的辅助函数
def to_one_hot(targets, num_classes):
    one_hot = torch.zeros(targets.size(0), num_classes, device=targets.device)  # 创建一个全零矩阵
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)  # 在目标位置设置为1
    return one_hot

# 测试函数
def test(net, testloader):
    net.eval()  # 设置模型为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总样本数
    criterion = SmoothCrossEntropyLoss(smoothing=0.1)  # 实例化平滑交叉熵损失函数

    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)  # 将输入和标签移动到 GPU
            targets_one_hot = to_one_hot(targets, num_classes=10)  # 转换为 one-hot 编码
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, targets_one_hot)  # 计算损失

            test_loss += loss.item()  # 累加测试损失
            _, predicted = outputs.max(1)  # 获取预测值
            total += targets.size(0)  # 累加样本数
            correct += predicted.eq(targets).sum().item()  # 累加正确预测数

            print(f'Batch {batch_idx+1}/{len(testloader)} | Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    acc = 100. * correct / total  # 计算准确率
    print(f'Test Accuracy: {acc:.3f}%')  # 打印测试准确率

if __name__ == '__main__':
    # 准备数据集和数据加载器
    trans_valid = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小到 256x256
        transforms.CenterCrop(224),  # 中心裁剪到 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=trans_valid)  # 加载测试集并应用变换
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)  # 创建 DataLoader

    # 初始化模型
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
    )

    # 加载检查点
    checkpoint = torch.load('./checkpoint/vit_cifar10_ckpt.pth')  # 加载模型检查点

    # 移除 `module.` 前缀
    state_dict = checkpoint['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 去除 `module.` 前缀
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)  # 加载模型参数
    net.eval()  # 设置模型为评估模式

    # 检查是否有可用的GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有 GPU，则使用 GPU，否则使用 CPU
    net = net.to(device)  # 将模型移动到 GPU

    # 执行测试
    test(net, testloader)  # 调用测试函数
