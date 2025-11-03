import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
# import numpy as np
# 查看系统可用字体
# for font in font_manager.fontManager.ttflist:
#     print(font.name)
# 设置中文字体
plt.rcParams['font.family'] = ['Heiti TC', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ======================================================================================
# 1. Noam学习率调度器实现
# ======================================================================================

class NoamScheduler:
    """
    Noam学习率调度器

    公式: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    重要：该方法中的step指的是每一个batch，而不是epoch。

    参数:
        optimizer: PyTorch优化器
        d_model: 模型隐藏层维度
        warmup_steps: 预热步数
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """更新学习率并执行优化步骤"""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def _get_lr(self) -> float:
        """根据当前步数计算学习率"""
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

    def get_current_lr(self) -> float:
        """获取当前学习率（不增加步数）"""
        return (self.d_model ** -0.5) * min(
            (self.step_num + 1) ** -0.5,
            (self.step_num + 1) * (self.warmup_steps ** -1.5)
        )


# ======================================================================================
# 2. 学习率曲线可视化
# ======================================================================================

def visualize_noam_schedule(d_model=512, warmup_steps=4000, total_steps=20000):
    """可视化Noam调度器的学习率变化曲线"""

    # 创建虚拟优化器和调度器
    dummy_model = nn.Linear(10, 10)
    optimizer = optim.Adam(dummy_model.parameters(), lr=0)
    scheduler = NoamScheduler(optimizer, d_model, warmup_steps)

    # 记录学习率
    steps = []
    lrs = []

    for step in range(1, total_steps + 1):
        scheduler.step_num = step
        lr = scheduler._get_lr()
        steps.append(step)
        lrs.append(lr)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(steps, lrs, linewidth=2, color='#2E86AB')
    plt.axvline(x=warmup_steps, color='red', linestyle='--',
                label=f'Warmup结束 (step={warmup_steps})')
    plt.xlabel('训练步数 (Step)', fontsize=12)
    plt.ylabel('学习率 (Learning Rate)', fontsize=12)
    plt.title(f'Noam学习率调度器 (d_model={d_model}, warmup={warmup_steps})',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    print(f"峰值学习率: {max(lrs):.6f}")
    print(f"峰值出现在第 {warmup_steps} 步")


# ======================================================================================
# 3. 实际训练示例：简单的序列预测任务
# ======================================================================================

class SimpleTransformer(nn.Module):
    """简化的Transformer模型用于演示"""

    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


def train_with_noam_scheduler():
    """使用Noam调度器训练模型的完整示例"""

    # 设置随机种子
    torch.manual_seed(42)

    # 超参数
    d_model = 512
    warmup_steps = 200
    num_epochs = 50
    batch_size = 32
    seq_len = 10

    # 创建模型
    model = SimpleTransformer(d_model=d_model)

    # # 创建优化器和调度器
    # optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = NoamScheduler(optimizer, d_model, warmup_steps)

    # Using PyTorch's LambdaLR for compatibility with optimizers
    optimizer = optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5)))

    # 损失函数
    criterion = nn.MSELoss()

    # 生成虚拟数据：预测sin函数
    def generate_data(batch_size, seq_len):
        x = torch.randn(batch_size, seq_len, 1)
        y = torch.sin(x * 2)
        return x, y

    # 训练循环
    losses = []
    lrs = []

    print("开始训练...\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 10

        for batch in range(num_batches):
            # 生成数据
            x, y = generate_data(batch_size, seq_len)

            # 前向传播
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率并优化

            epoch_loss += loss.item()

        # 记录
        avg_loss = epoch_loss / num_batches
        # current_lr = scheduler.get_current_lr()
        current_lr = optimizer.param_groups[0]['lr']
        losses.append(avg_loss)
        lrs.append(current_lr)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

    # 可视化训练过程
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(losses, linewidth=2, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 学习率曲线
    ax2.plot(lrs, linewidth=2, color='#F18F01')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('学习率变化曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n训练完成！")


# ======================================================================================
# 4. 运行示例
# ======================================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Noam学习率调度器演示")
    print("=" * 70)

    # 1. 可视化学习率曲线
    print("\n[1] 可视化学习率调度曲线")
    print("-" * 70)
    visualize_noam_schedule(d_model=512, warmup_steps=4000, total_steps=20000)

    # 2. 实际训练示例
    print("\n[2] 使用Noam调度器训练模型")
    print("-" * 70)
    train_with_noam_scheduler()
