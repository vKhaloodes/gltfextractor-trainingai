import json
import struct
from pathlib import Path

# ---------- مسارات الملفات ----------
GLTF_PATH = Path("for_edit.gltf")
BIN_PATH = Path("for_edit.bin")
x = input("Enter New Data: ")
JSON_PATH = Path(x)
OUT_BIN_PATH = Path("model1_updated.bin")
OUT_GLTF_PATH = Path("model1_updated.gltf")

# ---------- 1. تحميل بيانات vertices من JSON ----------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
vertices = data.get("vertices")
if not vertices:
    raise SystemExit("vertices null")

# فرد القوائم المتداخلة
if isinstance(vertices[0], (list, tuple)):
    vertices = [v for vertex in vertices for v in vertex]
print(f"Loaded {len(vertices)} float values from {JSON_PATH}")

# تحويلها إلى بايتات float32
vertices_bytes = b"".join(struct.pack("<f", float(v)) for v in vertices)
print(f"Packed {len(vertices_bytes)} bytes for vertices")

# ---------- 2. تحميل ملف glTF ----------
with open(GLTF_PATH, "r", encoding="utf-8") as f:
    gltf = json.load(f)

# العثور على أول POSITION accessor
pos_accessor_index = None
for mesh in gltf.get("meshes", []):
    for prim in mesh.get("primitives", []):
        if "POSITION" in prim.get("attributes", {}):
            pos_accessor_index = prim["attributes"]["POSITION"]
            break
    if pos_accessor_index is not None:
        break

if pos_accessor_index is None:
    raise SystemExit("gltf error")

print(f"Found POSITION accessor at index {pos_accessor_index}")

# جلب البيانات المرتبطة به
accessor = gltf["accessors"][pos_accessor_index]
buffer_view = gltf["bufferViews"][accessor["bufferView"]]
byte_offset = buffer_view.get("byteOffset", 0)
byte_length = buffer_view["byteLength"]

# تأكد أن حجم البايتات الجديد يطابق الحجم القديم
if len(vertices_bytes) != byte_length:
    raise SystemExit(f"bytelength error")

print(f"ℹReplacing POSITION data at offset {byte_offset}, length {byte_length}")

# ---------- 3. تعديل بيانات الباينري ----------
with open(BIN_PATH, "rb") as f:
    bin_data = bytearray(f.read())

bin_data[byte_offset:byte_offset + byte_length] = vertices_bytes

# حفظ نسخة جديدة من الباينري
with open(OUT_BIN_PATH, "wb") as f:
    f.write(bin_data)
print(f"Updated binary saved -> {OUT_BIN_PATH.resolve()}")

# ---------- 4. تحديث ملف glTF ليشير إلى الباينري الجديد ----------
gltf["buffers"][0]["uri"] = OUT_BIN_PATH.name
with open(OUT_GLTF_PATH, "w", encoding="utf-8") as f:
    json.dump(gltf, f, ensure_ascii=False, indent=2)

print(f"Updated glTF saved {OUT_GLTF_PATH.resolve()}")
print("Done")
