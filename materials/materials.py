import json
import numpy as np
import os

class MyMaterials:
    def __init__(self, path, real_dir):
        self.path = path
        self.real_dir = real_dir
        print("Materials Loader Ready ✅")

    def Load(self):
        # قراءة ملف glTF
        with open(self.path, "r") as f:
            gltf = json.load(f)

        # الوصول إلى أول primitive
        primitive = gltf["meshes"][0]["primitives"][0]

        # استخراج رقم المادة المرتبطة
        mat_index = primitive.get("material", None)
        if mat_index is None:
            print("⚠️ لا توجد مادة (Material) معرفة في هذا النموذج.")
            return None

        # الوصول إلى بيانات المادة
        material = gltf["materials"][mat_index]

        # استخراج خصائص PBR (اللون، اللمعان، الخشونة)
        pbr = material.get("pbrMetallicRoughness", {})

        base_color = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        metallic = pbr.get("metallicFactor", 1.0)
        roughness = pbr.get("roughnessFactor", 1.0)

        # محاولة استخراج مسار الخامة (الملف الصوري)
        tex_index = pbr.get("baseColorTexture", {}).get("index", None)
        texture_path = None
        if tex_index is not None:
            image_index = gltf["textures"][tex_index]["source"]
            image_uri = gltf["images"][image_index]["uri"]
            texture_path = os.path.join(self.real_dir, image_uri)

        # طباعة النتائج
        print("✅ بيانات المادة (Material):")
        print("Base Color:", base_color)
        print("Metallic Factor:", metallic)
        print("Roughness Factor:", roughness)
        if texture_path:
            print("Texture Path:", texture_path)
        else:
            print("⚠️ لا توجد صورة Texture مرفقة.")

        # تخزين المعلومات داخل الكلاس
        self.material = {
            "base_color": base_color,
            "metallic": metallic,
            "roughness": roughness,
            "texture_path": texture_path
        }

        return self.material
