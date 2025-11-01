import bpy

# مسار الملف (تأكد أنه صحيح ويحتوي على \\ في المسار)
filepath = r"C:\Users\Khaled 2\Desktop\gltfextractor\model.gltf"

# حذف كل شيء في المشهد أولاً (اختياري)
bpy.ops.wm.read_factory_settings(use_empty=True)

# استيراد GLTF
bpy.ops.import_scene.gltf(filepath=filepath)

print("Gltf test success")
