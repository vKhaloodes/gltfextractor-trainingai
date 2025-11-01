import os
import json
import glob
import random
import numpy as np
from typing import List


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# إعدادات عامة (عدّل المسارات هنا)
# =========================
DATA_DIR   = "data"  # مجلد JSON الموحّد
OUT_DIR    = "model"
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 200
Z_DIM = 128                 # طول الضوضاء
LR_G = 1e-4                 # تعلم المولّد
LR_D = 1e-4                 # تعلم المميّز
BETA1, BETA2 = 0.0, 0.9     # وفق WGAN-GP paper
N_CRITIC = 5                # عدد خطوات D لكل خطوة G
LAMBDA_GP = 10.0
NUM_WORKERS = 0             # على ويندوز خله 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1
SAVE_EVERY  = 10            # حفظ عينات كل X إيبُوك

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# Dataset
# =========================
class UnifiedJSONPointDataset(Dataset):
    """
    يتوقع ملفات JSON فيها: "vertices": [[x,y,z], ...]
    وكل الملفات بنفس عدد النقاط (N,3).
    """
    def __init__(self, folder: str):
        self.files = sorted(glob.glob(os.path.join(folder, "*.json")))
        if len(self.files) == 0:
            raise ValueError(f"No JSON files found in {folder}")
        with open(self.files[0], "r", encoding="utf-8") as f:
            d0 = json.load(f)
        self.N = len(d0.get("vertices", []))
        if self.N == 0:
            raise ValueError("First JSON has zero vertices or missing 'vertices' key")

        # التحقق من التوحيد
        for p in self.files:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            n = len(d.get("vertices", []))
            if n != self.N:
                raise ValueError(f"Inconsistent vertex count: {p} has {n}, expected {self.N}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        verts = np.asarray(d.get("vertices", []), dtype=np.float32)  # (N,3), نطاقها يُفترض [-1,1]
        return torch.from_numpy(verts)  # (N,3)


# =========================
# نماذج WGAN-GP (Point Cloud)
# =========================
class Generator(nn.Module):
    """
    يُحوّل z ∈ R^{Z_DIM} إلى سحابة نقاط (N,3) داخل [-1,1] باستخدام tanh.
    بسيط وفعال للبدء؛ تقدر تطوره لاحقًا (MLP أكبر / Conditioning / FoldingNet...).
    """
    def __init__(self, z_dim=128, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024), nn.ReLU(True),
            nn.Linear(1024, 2048), nn.ReLU(True),
            nn.Linear(2048, num_points * 3),
            nn.Tanh()  # ملائم لأن بياناتك مُطبّعة إلى [-1,1]
        )

    def forward(self, z):              # z: (B, Z_DIM)
        x = self.net(z)                # (B, N*3)
        return x.reshape(-1, self.num_points, 3)  # (B, N, 3)


class Discriminator(nn.Module):
    """
    مميّز بنمط PointNet صغير: Conv1d 1x1 + Global MaxPool + MLP
    يخرج قيمة سكر (realness score) بدون Sigmoid (WGAN loss).
    """
    def __init__(self, num_points=2048):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.relu  = nn.LeakyReLU(0.2, inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)  # لا Sigmoid في WGAN
        )

    def forward(self, x):  # x: (B, N, 3)
        x = x.transpose(1, 2)        # (B,3,N)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x)) # (B,256,N)
        x = torch.max(x, 2)[0]       # (B,256)
        return self.fc(x)            # (B,1)


# =========================
# أدوات التدريب والحفظ
# =========================
def gradient_penalty(D, real, fake):
    """WGAN-GP gradient penalty"""
    B = real.size(0)
    eps = torch.rand(B, 1, 1, device=real.device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    d_interp = D(interp)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B, N, 3)
    grads = grads.reshape(B, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def save_pointcloud_json(points: np.ndarray, out_path: str):
    """
    يحفظ نقاط (N,3) كـ JSON بسيط لاختبار التوليد.
    """
    obj = {"vertices": points.tolist()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_pointcloud_png(points: np.ndarray, out_path: str, title="generated"):
    """
    يحفظ رسم 3D بسيط للنقاط كصورة PNG (اختياري للمراجعة البصرية).
    """
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    # اضبط حدود المحاور لتكون [-1,1] لو بياناتك مطبعة
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def sample_and_save(G, epoch, num_samples, num_points):
    """
    يولّد عينات ويحفظها JSON + PNG
    """
    G.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, Z_DIM, device=DEVICE)
        fake = G(z).cpu().numpy()  # (B, N, 3)
    for i in range(num_samples):
        pts = fake[i]
        json_path = os.path.join(SAMPLES_DIR, f"epoch_{epoch:04d}_sample_{i}.json")
        png_path  = os.path.join(SAMPLES_DIR, f"epoch_{epoch:04d}_sample_{i}.png")
        save_pointcloud_json(pts, json_path)


# =========================
# التدريب
# =========================
def main():
    print(f"Device: {DEVICE}")
    dataset = UnifiedJSONPointDataset(DATA_DIR)
    N = dataset.N
    print(f"Found {len(dataset)} samples; unified points per mesh = {N}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    G = Generator(z_dim=Z_DIM, num_points=N).to(DEVICE)
    D = Discriminator(num_points=N).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    step = 0
    best_D = None

    for epoch in range(1, EPOCHS + 1):
        for real in loader:
            real = real.to(DEVICE)  # (B,N,3)

            # -------------------------
            # 1) Train Discriminator
            # -------------------------
            for _ in range(N_CRITIC):
                z = torch.randn(real.size(0), Z_DIM, device=DEVICE)
                fake = G(z).detach()

                d_real = D(real)              # (B,1)
                d_fake = D(fake)              # (B,1)
                gp = gradient_penalty(D, real, fake) * LAMBDA_GP
                d_loss = -(d_real.mean() - d_fake.mean()) + gp

                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()

            # -------------------------
            # 2) Train Generator
            # -------------------------
            z = torch.randn(real.size(0), Z_DIM, device=DEVICE)
            fake = G(z)
            g_loss = -D(fake).mean()

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            step += 1

        if epoch % PRINT_EVERY == 0:
            print(f"[Epoch {epoch:03d}/{EPOCHS}] "
                  f"D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

        # حفظ عينات بشكل دوري
        if epoch % SAVE_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            sample_and_save(G, epoch, num_samples=3, num_points=N)

        # حفظ نقاط تفتيش للموديلات
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            torch.save(G.state_dict(), os.path.join(OUT_DIR, f"G_epoch_{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(OUT_DIR, f"D_epoch_{epoch}.pth"))

    print("✅ التدريب اكتمل.")
    # توليد 10 عينات ختامية
    sample_and_save(G, EPOCHS, num_samples=10, num_points=N)
    print(f"📦 تم حفظ النماذج والعينات في: {OUT_DIR}")

if __name__ == "__main__":
    main()

#testing for generating vertices of the mesh

import matplotlib.pyplot as plt

# 🔹 المرحلة الرابعة: توليد سحابات نقاط جديدة باستخدام النموذج المدرب
print("\nبدء توليد نماذج جديدة باستخدام المولّد المدرب ...")

MODEL_PATH = os.path.join("model", "G_epoch_200.pth")
dir_extract = "extracted_data"
GENERATED_DIR = os.path.join(dir_extract, "generated_samples")
os.makedirs(GENERATED_DIR, exist_ok=True)

NUM_SAMPLES = 5      # عدد المجسمات التي تريد توليدها
NUM_POINTS = 129    # يجب أن يطابق عدد النقاط الموحد

try:
    G = Generator(z_dim=Z_DIM, num_points=NUM_POINTS).to(DEVICE)

    # تحميل أوزان النموذج
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    G.load_state_dict(state_dict)
    G.eval()

    print(f"تم تحميل مولّد النقاط من: {MODEL_PATH}")

    with torch.no_grad():
        z = torch.randn(NUM_SAMPLES, Z_DIM, device=DEVICE)
        fake_clouds = G(z).cpu().numpy()  # (B, N, 3)

    for i in range(NUM_SAMPLES):
        pts = fake_clouds[i]
        json_path = os.path.join(GENERATED_DIR, f"sample_{i}.json")
        img_path = os.path.join(GENERATED_DIR, f"sample_{i}.png")

        save_pointcloud_json(pts, json_path)
        save_pointcloud_png(pts, img_path, title=f"Generated Sample {i}")

        print(f"تم توليد النموذج: {json_path}")

    print("انتهت عملية التوليد بنجاح. راجع مجلد generated_samples.")
except Exception as e:
    print(f"❌ خطأ أثناء التوليد: {e}")
