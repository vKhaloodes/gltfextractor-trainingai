import json
import numpy as np

class MyVertices:
    def __init__(self, path, real_dir):
        self.path = path
        self.real_dir = real_dir
        print("Vertices Loader Ready ✅")

    def Load(self):
        # قراءة ملف glTF
        with open(self.path, "r") as f:
            gltf = json.load(f)

        # تحميل ملف bin المرتبط
        bin_uri = gltf["buffers"][0]["uri"]
        with open(self.real_dir + "/" + bin_uri, "rb") as f:
            bin_data = f.read()

        # استخراج أول Accessor (عادة يكون POSITION)
        accessor = gltf["accessors"][0]
        bufferview = gltf["bufferViews"][accessor["bufferView"]]

        # تحديد الموقع في الملف الثنائي
        byte_offset = bufferview.get("byteOffset", 0)
        byte_length = bufferview["byteLength"]

        # قص الجزء المطلوب من الملف الثنائي
        raw = bin_data[byte_offset: byte_offset + byte_length]

        # تحويله إلى أرقام Float32
        array = np.frombuffer(raw, dtype=np.float32)

        # إعادة تشكيله إلى (N, 3) إذا كان VEC3
        if accessor["type"] == "VEC3":
            array = array.reshape(accessor["count"], 3)

        print("First 5 vertices:")
        print(array[:5])

        # نحفظ المصفوفة داخل الكلاس
        self.vertices = array
        return array

