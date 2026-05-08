import os, shutil, copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ==========================================
# 0. 环境整理 (确保 images 文件夹存在且包含 jpg)
# ==========================================
if not os.path.exists("images"): os.makedirs("images")
for f in os.listdir("."):
    if f.endswith(".jpg") and "_" in f:
        shutil.move(f, os.path.join("images", f))

IMG_DIR = "images"
TVBN_FILE = "样品TVB-N含量对应表.xlsx"
DELTA_FILE = "特征值反应前后差值结果.xlsx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 环境就绪。设备: {device}")


# ==========================================
# 1. 数据预处理：将 20 个点展平为 1 个向量
# ==========================================
def prepare_fusion_data(tvbn_path, delta_path):
    # 读取原始 Excel
    tvbn_df = pd.read_excel(tvbn_path)
    delta_df = pd.read_excel(delta_path)

    # 清理列名空格
    tvbn_df.columns = tvbn_df.columns.str.strip()
    delta_df.columns = delta_df.columns.str.strip()

    # 处理差值表：将每个图片的 20 个点展平
    # 提取 R, G, B Diff 核心列
    diff_cols = ['R_Value_Diff', 'G_Value_Diff', 'B_Value_Diff']

    # 按 Image_Name 分组并拼接特征
    pivoted = delta_df.groupby('Image_Name').apply(
        lambda x: x.sort_values('Point_Number')[diff_cols].values.flatten()
    ).reset_index()
    pivoted.columns = ['原始文件名', 'Feature_Vector']

    # 与 TVB-N 标签表合并
    final_df = pd.merge(tvbn_df, pivoted, on='原始文件名').dropna(subset=['TVB-N含量(mg/100g)'])
    print(f"📊 数据合并成功！共有 {len(final_df)} 个样本，每个样本数值特征维度: {len(final_df.iloc[0]['Feature_Vector'])}")
    return final_df


# ==========================================
# 2. 增强型数据集类 (智能处理后缀)
# ==========================================
class MuttonDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 后缀自适应逻辑：尝试加载 .jpg
        img_name = str(row["原始文件名"]).replace(".bmp", ".jpg").replace(".BMP", ".jpg")
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # 最后的备选：若找不到，生成一张全黑图防止崩溃
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform: image = self.transform(image)

        # 提取数值特征和标签
        features = torch.tensor(row['Feature_Vector'], dtype=torch.float32) / 255.0
        target = torch.tensor(row["TVB-N含量(mg/100g)"], dtype=torch.float32)

        return image, features, target


# ==========================================
# 3. 融合网络 (CNN + MLP)
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, num_numeric):
        super(FusionNet, self).__init__()
        # 图像分支
        self.cnn = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 数值分支
        self.mlp = nn.Sequential(nn.Linear(num_numeric, 64), nn.ReLU(), nn.Dropout(0.2))
        # 融合层
        self.regressor = nn.Sequential(
            nn.Linear(1280 + 64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, img, num):
        x1 = self.pool(self.cnn(img)).view(img.size(0), -1)
        x2 = self.mlp(num)
        return self.regressor(torch.cat((x1, x2), dim=1))


# ==========================================
# 4. 执行训练与独立测试评估
# ==========================================
def main():
    # 1. 数据准备
    final_df = prepare_fusion_data(TVBN_FILE, DELTA_FILE)
    num_numeric = len(final_df.iloc[0]['Feature_Vector'])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---------------------------------------------------------
    # 核心修改：三划分法 (训练集 64% | 验证集 16% | 测试集 20%)
    # ---------------------------------------------------------
    # 第一步：切分出 20% 作为绝对独立的测试集
    train_val_idx, test_idx = train_test_split(range(len(final_df)), test_size=0.2, random_state=42)
    # 第二步：将剩下的 80% 再切分 (80%里的80%做训练，20%做验证)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)

    print(f"🗂️ 数据集划分：训练集 {len(train_idx)} 个 | 验证集 {len(val_idx)} 个 | 测试集 {len(test_idx)} 个")

    train_ds = Subset(MuttonDataset(final_df, IMG_DIR, transform), train_idx)
    val_ds = Subset(MuttonDataset(final_df, IMG_DIR, transform), val_idx)
    test_ds = Subset(MuttonDataset(final_df, IMG_DIR, transform), test_idx)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = FusionNet(num_numeric).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.HuberLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_r2 = -float('inf')
    best_model_weights = None

    print("🚀 启动模型训练 (监控验证集)...")
    for epoch in range(100):
        # --- 训练阶段 ---
        model.train()
        for imgs, nums, labels in train_loader:
            imgs, nums, labels = imgs.to(device), nums.to(device), labels.view(-1, 1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, nums), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # --- 验证阶段 ---
        model.eval()
        val_p, val_t = [], []
        with torch.no_grad():
            for imgs, nums, labels in val_loader:
                out = model(imgs.to(device), nums.to(device))
                val_p.extend(out.cpu().numpy().flatten())
                val_t.extend(labels.numpy())

        cur_val_r2 = r2_score(val_t, val_p)

        # 记录验证集表现最好的模型权重
        if cur_val_r2 > best_val_r2:
            best_val_r2 = cur_val_r2
            # 使用 deepcopy 深度拷贝权重保存到内存，防止被后续 epoch 覆盖
            best_model_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | 验证集 R²: {cur_val_r2:.4f}")

    print(f"\n✅ 训练阶段结束！验证集最高 R²: {best_val_r2:.4f}")

    # ==========================================
    # 5. 终极盲测 (加载最好模型，在测试集上预测)
    # ==========================================
    print("🎯 正在加载最强模型对【独立测试集】进行终极预测...")
    model.load_state_dict(best_model_weights)
    model.eval()

    test_p, test_t = [], []
    with torch.no_grad():
        for imgs, nums, labels in test_loader:
            out = model(imgs.to(device), nums.to(device))
            test_p.extend(out.cpu().numpy().flatten())
            test_t.extend(labels.numpy())

    final_test_r2 = r2_score(test_t, test_p)
    print(f"🔥 独立测试集（模型从未见过的数据）最终 R²: {final_test_r2:.4f}")

    # 绘图：只画测试集的真实结果
    plt.figure(figsize=(8, 6))
    plt.scatter(test_t, test_p, color='blue', alpha=0.6, label='Predicted vs True')

    # 画一条完美的 y=x 参考线
    min_val = min(min(test_t), min(test_p))
    max_val = max(max(test_t), max(test_p))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')

    plt.title(f"True Test Set Prediction (R² = {final_test_r2:.4f})")
    plt.xlabel("True TVB-N (mg/100g)")
    plt.ylabel("Predicted TVB-N (mg/100g)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()