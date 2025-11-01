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

`main.py`

### 4) التطبيع والتوحيد

`Normalizer/normalizer.py`


### 5) التدريب (WGAN‑GP) والتوليد

`train.py`

### 7) أوامر التشغيل السريعة

```bash
# 1) تثبيت المتطلبات
pip install -r requirements.txt

# 2) استخراج GLTF → JSON (raw)
python main.py

# 3) التطبيع والتوحيد
python Normalizer/normalizer.py
python autounifier/unifier.py

# 4) التدريب والتوليد
python train.py
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




