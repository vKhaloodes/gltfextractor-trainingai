import json
import numpy as np

class MyIndices:
    def __init__(self, path, real_dir):
        self.path = path
        self.real_dir = real_dir
        print("Indices Loaded")

    def Load(self):

        with open(self.path, "r") as f:
            gltf = json.load(f)

        # تحميل ملف bin المرتبط
        bin_uri = gltf["buffers"][0]["uri"]
        with open(self.real_dir + "/" + bin_uri, "rb") as f:
            bin_data = f.read()

        # الوصول إلى أول primitive
        primitive = gltf["meshes"][0]["primitives"][0]

        # استخراج بيانات indices
        indices_accessor = gltf["accessors"][primitive["indices"]]
        indices_view = gltf["bufferViews"][indices_accessor["bufferView"]]

        offset = indices_view.get("byteOffset", 0)
        length = indices_view["byteLength"]
        raw = bin_data[offset:offset + length]

        dtype_map = {
            5121: np.uint8,
            5123: np.uint16,
            5125: np.uint32
        }
        dtype = dtype_map[indices_accessor["componentType"]]
        indices = np.frombuffer(raw, dtype=dtype)

        # كل 3 أرقام تمثل مثلث واحد
        faces = indices.reshape(-1, 3)
        print("ول 5 وجوه:")
        print(faces[:5])

        return faces