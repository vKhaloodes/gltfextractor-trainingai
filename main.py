import json
import numpy as np
import os

from object_main.Indices import MyIndices
from object_main.vertices import MyVertices
from object_main.normals import MyNormals
from textures.textures import MyTextures
from materials.materials import MyMaterials
from Normalizer.normalizer import MyNormalizer
from autounifier.autounifer import MyAutoUnifier
from topology import TopologyChecker


class GLTFExtractor:
    def __init__(self, model_path, real_dir, dir_extract, normalize=True, target_points=2048):
        self.model_path = model_path
        self.real_dir = real_dir
        self.dir_extract = dir_extract
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.normalize = normalize
        self.target_points = target_points
        print(f"\nGLTFExtractor Ready — model: {self.model_name}")

    def Extract(self):
        # 1️⃣ تحميل كل أنواع البيانات
        vertices = MyVertices(self.model_path, self.real_dir).Load()
        indices = MyIndices(self.model_path, self.real_dir).Load()
        normals = MyNormals(self.model_path, self.real_dir).Load()
        uvs = MyTextures(self.model_path, self.real_dir).Load()
        materials = MyMaterials(self.model_path, self.real_dir).Load()

        # 2️⃣ تحويل numpy arrays إلى lists (حتى يمكن تخزينها في JSON)
        def safe_convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        data = {
            "model_name": self.model_name,
            "vertices": safe_convert(vertices),
            "indices": safe_convert(indices),
            "normals": safe_convert(normals),
            "uvs": safe_convert(uvs),
            "materials": safe_convert(materials)
        }

        # 3️⃣ حفظ البيانات في ملف JSON
        os.makedirs(self.dir_extract, exist_ok=True)
        output_path = os.path.join(self.dir_extract, f"{self.model_name}_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data saved success\n{output_path}")

        # 4️⃣ تطبيق Normalization + Resampling (اختياري)
        if self.normalize:
            print(f"⚙️ بدء عملية التطبيع والتوحيد على {self.model_name} ...")
            normalizer = MyNormalizer(output_path, target_points=self.target_points)
            normalized_path = normalizer.Process()
            print(f"The File is normalized \n{normalized_path}")

        return output_path


# 🔹 مثال استخدام — معالجة جميع ملفات .gltf في المجلد
if __name__ == "__main__":
    real_dir = r"C:\Users\Khaled 2\Desktop\gltfextractor\objects"
    dir_extract = r"C:\Users\Khaled 2\Desktop\gltfextractor\extracted_data"

    # 🔁 نبحث عن كل ملفات GLTF داخل مجلد objects
    gltf_files = [f for f in os.listdir(real_dir) if f.lower().endswith(".gltf")]

    if not gltf_files:
        print("No Folders Found.gltf في المجلد المحدد.")
    else:
        print(f"Found {len(gltf_files)} ملف .gltf\n")

        # 🔹 المرحلة الأولى: استخراج + تطبيع لكل ملف
        for file_name in gltf_files:
            model_path = os.path.join(real_dir, file_name)
            try:
                extractor = GLTFExtractor(model_path, real_dir, dir_extract, normalize=True, target_points=2048)
                extractor.Extract()
            except Exception as e:
                print(f"Error while working{file_name}: {e}")

        # 🔹 المرحلة الثانية: توحيد جميع الملفات بعد الانتهاء
        print("unifiying stage")
        unified_output = os.path.join(dir_extract, "unified")
        try:
            unifier = MyAutoUnifier(dir_extract, unified_output)
            unifier.unify_all()
            print("unified success")
        except Exception as e:
            print(f"Error Happened")

        # 🔹 المرحلة الثالثة: فحص التوبولوجي لجميع الملفات الموحدة
        print("\nFixed all topology.")
        try:
            checker = TopologyChecker(unified_output)
            checker.compare_all()
            print("\nTopology Complete")
        except Exception as e:
            print(f"Error at topology")
