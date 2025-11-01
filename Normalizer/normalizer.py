import os
import json
import numpy as np

class MyNormalizer:
    def __init__(self, file_path, target_points=2048):
        """
        🔹 file_path: مسار ملف JSON الخاص بالمجسم
        🔹 target_points: عدد النقاط بعد إعادة أخذ العينات (Resampling)
        """
        self.file_path = file_path
        self.target_points = target_points
        self.output_dir = os.path.join(os.path.dirname(file_path), "normalized_single")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📦 Normalizer Ready → {os.path.basename(file_path)}")

    # ========== 🔹 دالة Resampling ==========
    def resample_vertices(self, vertices):
        """إعادة أخذ عينات لتوحيد عدد النقاط"""
        vertices = np.array(vertices, dtype=np.float32)
        count = len(vertices)
        if count == 0:
            return np.zeros((self.target_points, 3), dtype=np.float32)

        if count >= self.target_points:
            idx = np.random.choice(count, self.target_points, replace=False)
        else:
            idx = np.random.choice(count, self.target_points, replace=True)

        return vertices[idx]

    # ========== 🔹 دالة Normalization ==========
    def normalize_vertices(self, vertices):
        """تطبيع الإحداثيات إلى [-1, 1]"""
        vertices = np.array(vertices, dtype=np.float32)
        if vertices.size == 0:
            return vertices

        # نحط المركز عند (0,0,0)
        center = np.mean(vertices, axis=0)
        vertices -= center

        # نطبع الحجم بحيث يصير أقصى طول = 1
        scale = np.max(np.linalg.norm(vertices, axis=1))
        if scale > 0:
            vertices /= scale

        return vertices

    # ========== 🔹 دالة التشغيل ==========
    def Process(self):
        """ينفّذ العملية الكاملة: قراءة → Resample → Normalize → حفظ"""
        if not os.path.exists(self.file_path):
            print(f"⚠️ الملف غير موجود: {self.file_path}")
            return

        print(f"🔹 معالجة الملف: {os.path.basename(self.file_path)}")

        # قراءة الملف
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # جلب الـ vertices
        vertices = np.array(data.get("vertices", []), dtype=np.float32)

        # تنفيذ الخطوات
        resampled = self.resample_vertices(vertices)
        normalized = self.normalize_vertices(resampled)

        # تحديث البيانات
        data["vertices"] = normalized.tolist()
        data["normalized"] = True
        data["target_points"] = self.target_points

        # حفظ الملف الموحد
        output_path = os.path.join(self.output_dir, os.path.basename(self.file_path))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"تم حفظ النسخة الموحدة في:\n{output_path}")
        return output_path
