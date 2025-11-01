import json
import numpy as np

class MyTextures:
    def __init__(self, path, real_dir):
        self.path = path
        self.real_dir = real_dir
        print("Textures Loader Ready ✅")

    def Load(self):
        # قراءة ملف glTF
        with open(self.path, "r") as f:
            gltf = json.load(f)

        # تحميل ملف bin المرتبط
        bin_uri = gltf["buffers"][0]["uri"]
        with open(self.real_dir + "/" + bin_uri, "rb") as f:
            bin_data = f.read()

        # الوصول إلى أول primitive
        primitive = gltf["meshes"][0]["primitives"][0]

        # التأكد من وجود TEXCOORD_0
        if "TEXCOORD_0" not in primitive["attributes"]:
            print("⚠️ لا توجد إحداثيات Texture (TEXCOORD_0) في هذا النموذج.")
            return None

        # استخراج Accessor الخاص بإحداثيات الـ UV
        tex_accessor = gltf["accessors"][primitive["attributes"]["TEXCOORD_0"]]
        tex_view = gltf["bufferViews"][tex_accessor["bufferView"]]

        # تحديد الموقع داخل الملف الثنائي
        byte_offset = tex_view.get("byteOffset", 0)
        byte_length = tex_view["byteLength"]

        # قص الجزء المطلوب من الملف الثنائي
        raw = bin_data[byte_offset: byte_offset + byte_length]

        # تحويله إلى أرقام Float32 (u, v)
        array = np.frombuffer(raw, dtype=np.float32)

        # إعادة تشكيله إلى (N, 2) لأن كل UV يحتوي على عنصرين
        if tex_accessor["type"] == "VEC2":
            array = array.reshape(tex_accessor["count"], 2)

        print("First 5 Texture UV coordinates:")
        print(array[:5])

        # حفظ المصفوفة داخل الكلاس
        self.uvs = array
        return array
