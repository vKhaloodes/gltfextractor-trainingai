import os
import json
import numpy as np
import matplotlib.pyplot as plt

# مسار المجلد اللي يحتوي ملفات JSON
data_dir = r"C:\Users\Khaled 2\Desktop\gltfextractor\extracted_data"

# نقرأ كل الملفات داخل المجلد
json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

all_values = []  # لتجميع كل القيم الرقمية من جميع الملفات

for file in json_files:
    path = os.path.join(data_dir, file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # نحاول نأخذ فقط القيم الرقمية (مثل vertices, normals, uvs)
    for key, value in data.items():
        if isinstance(value, list):
            arr = np.array(value, dtype=np.float32)
            if arr.size > 0 and arr.ndim > 1:  # نتأكد إنها مصفوفة حقيقية
                # ندمجها كلها في مصفوفة واحدة
                all_values.extend(arr.flatten())

# نحول القائمة إلى مصفوفة NumPy
all_values = np.array(all_values, dtype=np.float32)

# نحسب المتوسط (mean)
mean_val = np.mean(all_values)
print(f"✅ Mean of all extracted data: {mean_val:.4f}")

std = np.std(all_values)
print("Standard deviation:", std)

# نرسم Histogram (bins)
plt.figure(figsize=(8,5))
plt.hist(all_values, bins=50, color="skyblue", edgecolor="black")
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f"Mean = {mean_val:.3f}")
plt.title("Distribution of 3D Model Data (Vertices, Normals, UVs, etc.)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.show()