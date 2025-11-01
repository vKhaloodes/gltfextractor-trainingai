import os
import json
import numpy as np

class MyAutoUnifier:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📦 AutoUnifier Ready — source: {self.input_dir}")

    def resize_array(self, arr, target_len):
        """تقص أو تكرّر عناصر مصفوفة لتطابق الطول المطلوب"""
        arr = np.array(arr, dtype=np.float32)
        n = len(arr)
        if n == 0:
            return np.zeros((target_len, 3), dtype=np.float32)
        if n > target_len:
            idx = np.random.choice(n, target_len, replace=False)
        else:
            idx = np.random.choice(n, target_len, replace=True)
        return arr[idx]

    def get_max_counts(self, files):
        """يقرأ جميع الملفات ويكتشف أكبر عدد لكل نوع بيانات"""
        max_counts = {"vertices": 0, "indices": 0, "normals": 0, "uvs": 0}
        for file in files:
            path = os.path.join(self.input_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key in max_counts.keys():
                if key in data and isinstance(data[key], list):
                    max_counts[key] = max(max_counts[key], len(data[key]))
        return max_counts

    def unify_all(self):
        """يوحد جميع الملفات على نفس عدد العناصر (حسب الأكبر)"""
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        if not files:
            print("⚠️ لا توجد ملفات JSON في المجلد.")
            return

        print(f"📂 تم العثور على {len(files)} ملف JSON.")
        max_counts = self.get_max_counts(files)
        print(f"\n📊 القيم القصوى المكتشفة:")
        for k, v in max_counts.items():
            print(f"   {k}: {v}")

        for file in files:
            path = os.path.join(self.input_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key, target_len in max_counts.items():
                if key in data:
                    data[key] = self.resize_array(data[key], target_len).tolist()

            output_path = os.path.join(self.output_dir, file)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            print(f"✅ تم توحيد {file}")

        print("\n🎯 جميع الملفات أصبحت بنفس عدد العناصر!")



# 🔹 مثال استخدام
if __name__ == "__main__":
    input_dir = r"C:\Users\Khaled 2\Desktop\gltfextractor\extracted_data"
    output_dir = os.path.join(input_dir, "unified")

    unifier = MyAutoUnifier(input_dir, output_dir)
    unifier.unify_all()
