import os
import json
import numpy as np

class TopologyChecker:
    def __init__(self, dir_path):
        """
        🔹 dir_path: Get Json Path
        """
        self.dir_path = dir_path
        self.files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        if not self.files:
            raise ValueError("⚠️ لا توجد ملفات JSON في المجلد المحدد.")
        print(f"found path")

    def load_mesh(self, file_name):
        """Getting Vertices and indices"""
        path = os.path.join(self.dir_path, file_name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vertices = np.array(data.get("vertices", []), dtype=np.float32)
        indices = np.array(data.get("indices", []), dtype=np.int32)
        return vertices, indices

    def compare_two(self, file1, file2):
        """Comparing"""
        v1, i1 = self.load_mesh(file1)
        v2, i2 = self.load_mesh(file2)

        print(f"Comparing")
        print(f" {file1}")
        print(f"  {file2}")

        same_vertex_count = len(v1) == len(v2)
        same_index_count = len(i1) == len(i2)
        same_indices = np.array_equal(i1, i2)

        print(f"vertices: {len(v1)} vs {len(v2)}")
        print(f"indices: {len(i1)} vs {len(i2)}")

        if same_vertex_count and same_index_count and same_indices:
            print("topoly all same")
            return True
        else:
            print("topology different")
            if not same_vertex_count:
                print("vertices different")
            if not same_index_count:
                print("index count different")
            if not same_indices:
                print("indices different")
            return False

    def compare_all(self):
        """Comparing system"""
        n = len(self.files)
        if n < 2:
            print("2 files needed")
            return

        print("starting comparing.")
        results = {}

        for i in range(n):
            for j in range(i + 1, n):
                file1, file2 = self.files[i], self.files[j]
                result = self.compare_two(file1, file2)
                results[(file1, file2)] = result

        print("Result Summary")
        for pair, same in results.items():
            status = "Same" if same else "Different"
            print(f"   {pair[0]} ↔ {pair[1]} → {status}")

        return results
