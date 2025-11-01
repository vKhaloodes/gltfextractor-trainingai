import json
import numpy as np

class MyNormals:
    def __init__(self, path, real_dir):
        self.path = path
        self.real_dir = real_dir
        print("Normals Loader Ready ✅")

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

        # التأكد من وجود NORMAL
        if "NORMAL" not in primitive["attributes"]:
            print("⚠️ لا توجد بيانات Normals في هذا النموذج.")
            return None

        # استخراج Accessor الخاص بـ NORMAL
        normal_accessor = gltf["accessors"][primitive["attributes"]["NORMAL"]]
        normal_view = gltf["bufferViews"][normal_accessor["bufferView"]]

        # تحديد الموقع داخل الملف الثنائي
        byte_offset = normal_view.get("byteOffset", 0)
        byte_length = normal_view["byteLength"]

        # قص الجزء المطلوب من الملف الثنائي
        raw = bin_data[byte_offset: byte_offset + byte_length]

        # تحويله إلى أرقام Float32
        array = np.frombuffer(raw, dtype=np.float32)

        # إعادة تشكيله إلى (N, 3) لأن كل normal يتكون من (x, y, z)
        if normal_accessor["type"] == "VEC3":
            array = array.reshape(normal_accessor["count"], 3)

        print("First Normals:")
        print(array[:5])

        # نحفظها داخل الكلاس
        self.normals = array
        return array
