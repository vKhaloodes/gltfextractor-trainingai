# GLTFExtractor: 3D Point Cloud Extraction, Training, and Generation System

> **من تطوير: خالد الزهراني – AKA CAKhaled**

GLTFExtractor هو خطّ معالجة (Pipeline) معياري مبنيّ ببايثون لاستخراج البيانات الهندسية والمادية من ملفات `.gltf`، تحويلها إلى مجموعات بيانات منظّمة، تدريب نموذج توليدي على سحابات النقاط (Point Clouds) باستخدام **WGAN‑GP**، ثم توليد عينات 3D جديدة.

---

## المزايا الأساسية

* **استخراج GLTF شامل**: vertices, indices, normals, uvs, materials.
* **تطبيع وإعادة أخذ عينات**: توحيد نطاق الإحداثيات وعدد النقاط لكل عيّنة.
* **تجميع ملفات متعددة**: تدقيق التوافق الطوبولوجي وتوحيدها في مجلد واحد للتدريب.
* **تدريب WGAN‑GP**: نماذج Generator/Discriminator ملائمة لسحابات نقاط.
* **توليد عينات جديدة**: حفظ النتائج بصيغة JSON وصور معاينة PNG.

---

## بنية الوحدات (Modules)

| Module                      | Responsibility                          |
| --------------------------- | --------------------------------------- |
| `MyVertices`                | استخراج الإحداثيات من GLTF              |
| `MyIndices`                 | استخراج بيانات الوجوه (indices)         |
| `MyNormals`                 | استخراج المتجهات العمودية               |
| `MyTextures`                | استخراج إحداثيات الـ UV                 |
| `MyMaterials`               | استخراج معلومات الخامات                 |
| `MyNormalizer`              | تطبيع/إعادة أخذ عينات لسحابة النقاط     |
| `MyAutoUnifier`             | توحيد الملفات والتأكد من نفس عدد النقاط |
| `TopologyChecker`           | فحص الاتساق الطوبولوجي                  |
| `Generator / Discriminator` | شبكات GAN للتدريب والتوليد              |

---

## تدفق البيانات (Data Flow)

1. **GLTF → JSON**: استخراج كل مكوّن وحفظه في JSON منظم.
2. **Normalization & Resampling**: تحويل كل نموذج لنطاق موحد وعدد نقاط ثابت.
3. **Unification**: التأكد من الاتساق ووضع الناتج في مجلد `data/` الجاهز للتدريب.
4. **Training (WGAN‑GP)**: تدريب المولد والمميز على سحابات النقاط الموحدة.
5. **Generation**: تمرير ضوضاء عشوائية للمولد لإنتاج سحابات جديدة.

---

## نظام مبدئي

> هذا المشروع في نسخته الأولية ويعتبر **قابلًا للتطوير** بإضافة دعم لشبكات أحدث، مثل PointNet++ أو Transformer-based Point Clouds.

### مكونات GAN

* **Generator**: يقوم بتوليد سحابات نقاط ثلاثية الأبعاد بناءً على ضوضاء مُدخلة.
* **Discriminator**: يميّز بين سحابة نقاط حقيقية وأخرى مولدة.

> يعتمد المشروع على GAN (WGAN-GP) كمنهج توليدي أساسي.

### ملاحظة مهمة حول التدريب

> في هذه النسخة الأولية من المشروع، لا يتم تنفيذ خطوات **Normalization** أو **Unification** أثناء التدريب. يتم تدريب النموذج مباشرة على البيانات المستخرجة من ملفات GLTF (بعد تحويلها إلى JSON فقط)، وذلك لتبسيط خط العمل في مرحلة الـ Prototype.

## نظام مبدئي (MVP System) (Prototype)

> هذا المشروع في نسخته الأولية ويعتبر **قابلًا للتطوير** بإضافة دعم لشبكات أحدث، مثل PointNet++ أو Transformer-based Point Clouds.

### مكونات GAN

* **Generator**: يقوم بتوليد سحابات نقاط ثلاثية الأبعاد بناءً على ضوضاء مُدخلة.
* **Discriminator**: يميّز بين سحابة نقاط حقيقية وأخرى مولدة.

> يعتمد المشروع على GAN (WGAN-GP) كمنهج توليدي أساسي. (MVP System)

> الهدف: تشغيل نسخة أولية من النظام من الاستخراج إلى التوليد بأقل تعقيد ممكن.

### 1) هيكل المشروع المقترح

```
GLTFExtractor/
├─ gltf/
│  └─ inputs/                 # ضع هنا ملفات .gltf
├─ data/                      # مخرجات JSON الموحدة للتدريب
├─ model/                     # أوزان التدريب + عينات التوليد
│  └─ samples/
├─ src/
│  ├─ extractor.py            # استخراج من GLTF → JSON
│  ├─ normalizer.py           # تطبيع/إعادة أخذ عينات
│  ├─ unifier.py              # توحيد الملفات وتحقق الطوبولوجيا
│  ├─ train_wgan_gp.py        # تدريب WGAN-GP على سحابات النقاط
│  ├─ generate.py             # توليد عينات بعد التدريب
│  └─ utils_io.py             # أدوات مساعدة (قراءة/كتابة JSON/PNG)
├─ requirements.txt
└─ README.md
```

### 2) المتطلبات

`requirements.txt`

```
numpy
trimesh
pygltflib
matplotlib
scipy
scikit-image
scikit-learn
torch
bpy
```

### 3) استخراج بسيط من GLTF → JSON

`src/extractor.py`

```python
from pathlib import Path
import json
import numpy as np
import trimesh

IN_DIR = Path("gltf/inputs")
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mesh_to_point_cloud(mesh: trimesh.Trimesh, target_points=2048):
    # أخذ نقاط من السطح (sampling) إن لم تكن vertices كافية أو متوازنة
    if hasattr(mesh, 'sample'):
        pts = mesh.sample(target_points)
    else:
        # fallback على vertices (قد تكون أقل جودة كتوزيع)
        pts = mesh.vertices
        if pts.shape[0] > target_points:
            idx = np.random.choice(pts.shape[0], target_points, replace=False)
            pts = pts[idx]
        elif pts.shape[0] < target_points:
            extra = np.random.choice(pts.shape[0], target_points - pts.shape[0], replace=True)
            pts = np.vstack([pts, pts[extra]])
    return pts.astype(np.float32)


def process_file(path: Path, target_points=2048):
    scene = trimesh.load(path, force='scene')
    pts_all = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            pts = mesh_to_point_cloud(geom, target_points=target_points)
            pts_all.append(pts)
    if not pts_all:
        return None
    pts = np.concatenate(pts_all, axis=0)
    # إعادة أخذ عيّنة أخيرة للوصول لعدد نقاط موحد
    if pts.shape[0] != target_points:
        idx = np.random.choice(pts.shape[0], target_points, replace=True)
        pts = pts[idx]
    return {"vertices": pts.tolist()}


def main():
    for f in IN_DIR.glob("*.gltf"):
        obj = process_file(f, target_points=2048)
        if obj is None:
            print(f"[!] No mesh in {f}")
            continue
        out = OUT_DIR / (f.stem + ".json")
        with open(out, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, ensure_ascii=False, indent=2)
        print(f"[OK] → {out}")

if __name__ == "__main__":
    main()
```

### 4) التطبيع والتوحيد

`src/normalizer.py`

```python
from pathlib import Path
import json
import numpy as np

RAW_DIR = Path("data/raw")
NORM_DIR = Path("data/norm")
NORM_DIR.mkdir(parents=True, exist_ok=True)
TARGET_POINTS = 2048


def normalize_unit_cube(pts: np.ndarray):
    # إزاحة لمركز الكتلة ثم تحجيم ليكون ضمن [-1,1]
    cen = pts.mean(axis=0, keepdims=True)
    pts = pts - cen
    scale = np.abs(pts).max()
    if scale > 0:
        pts = pts / scale
    return pts

for f in RAW_DIR.glob("*.json"):
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    pts = np.array(data["vertices"], dtype=np.float32)
    # تأكيد عدد النقاط
    if pts.shape[0] != TARGET_POINTS:
        idx = np.random.choice(pts.shape[0], TARGET_POINTS, replace=True)
        pts = pts[idx]
    pts = normalize_unit_cube(pts)
    out = NORM_DIR / f.name
    with open(out, "w", encoding="utf-8") as fp:
        json.dump({"vertices": pts.tolist()}, fp, ensure_ascii=False, indent=2)
    print("[OK] →", out)
```

`src/unifier.py`

```python
from pathlib import Path
import json

NORM_DIR = Path("data/norm")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ببساطة انسخ/تحقق؛ في نسخة MVP نفترض جميع الملفات موحّدة
for f in NORM_DIR.glob("*.json"):
    with open(f, "r", encoding="utf-8") as fp:
        d = json.load(fp)
    if "vertices" not in d or not isinstance(d["vertices"], list):
        print("[SKIP] invalid:", f)
        continue
    out = OUT_DIR / f.name
    with open(out, "w", encoding="utf-8") as fo:
        json.dump(d, fo, ensure_ascii=False, indent=2)
    print("[OK] unified →", out)
```

### 5) التدريب (WGAN‑GP)

`src/train_wgan_gp.py`

```python
# أعِد استخدام كودك الحالي للتدريب كما هو تقريبًا، مع ضبط المسارات إلى data/ و model/
# يُنصح بتحديد Z_DIM و TARGET_POINTS بما يتماشى مع normalizer/extractor
```

### 6) التوليد بعد التدريب

`src/generate.py`

```python
# أعِد استخدام دالة التوليد لديك (load_state_dict → z → G(z) → حفظ JSON/PNG)
```

### 7) أوامر التشغيل السريعة

```bash
# 1) تثبيت المتطلبات
pip install -r requirements.txt

# 2) استخراج GLTF → JSON (raw)
python src/extractor.py

# 3) التطبيع والتوحيد
python src/normalizer.py
python src/unifier.py

# 4) التدريب
python src/train_wgan_gp.py

# 5) التوليد
python src/generate.py
```

---

## تنسيق البيانات (JSON)

```json
{
  "vertices": [
    [0.12, -0.56, 0.87],
    [-0.33, 0.44, -0.21]
  ]
}
```

---

## طريقة الاستخدام (Usage)

1. **ضع ملفات GLTF** في مجلد `objects/` داخل مشروعك.

   * ملاحظة: يُفضل أن تكون جميع المجسمات متشابهة في نوع الشكل والسمات (على الأقل في عدد النقاط بعد الاستخراج).

2. **شغّل البرنامج الأساسي لاستخراج البيانات**:

   ```bash
   python main.py
   ```

   * سيتم استخراج بيانات كل مجسّم في شكل JSON داخل مجلد `extracted_data/`.

3. **انتقل إلى تدريب النموذج**:

   ```bash
   python train.py
   ```

   * سيستخدم بيانات JSON المستخرجة لتدريب نموذج WGAN-GP.

4. **عرض النتائج**:

   * بعد التدريب، ستجد عينات مُولّدة داخل `model/samples/` أو أي مسار محدد في سكربت التوليد.

---

## ملاحظات عملية

* تأكد من أن **TARGET_POINTS** موحّد عبر `extractor.py` و `normalizer.py` و `train_wgan_gp.py` و `generate.py`.
* عيّنات PNG اختيارية للمراجعة البصرية، ويمكن إزالتها في البيئات الخالية من الواجهة الرسومية.
* لاحقًا أضف **Chamfer/EMD** للتقييم، وSurface Reconstruction (Poisson/Ball Pivoting) لإنتاج Mesh.

---

## الإشادة (Credits)

* **التطوير**: خالد الزهراني – CAKHALED
* **الفكرة والبنية**: مستوحاة من ممارسات أبحاث 3D ML الحديثة (PointNet/WGAN‑GP)
