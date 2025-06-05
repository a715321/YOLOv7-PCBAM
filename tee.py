import os
from collections import Counter
import matplotlib.pyplot as plt




# 資料夾路徑
folder_path = "/home/yhshih/yolov7/NEUDET/train/labels"  # 替換為你的資料夾路徑

# 定義類別對應關係
categories = [
    'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'
]
class_mapping = {
    "0": "crazing",
    "1": "inclusion",
    "2": "patches",
    "3": "pitted_surface",
    "4": "rolled-in_scale",
    "5": "scratches",
}

# 初始化類別計數器
class_counter = Counter()

# 遍歷資料夾中的所有檔案
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):  # 只處理 .txt 文件
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            for line in file:
                class_label = line.split()[0]  # 提取每行的第一個欄位作為類別
                class_counter[class_label] += 1

# 按 class 排序並映射名稱
sorted_classes = sorted(class_counter.items())  # 排序字典
categories = [class_mapping[item[0]] for item in sorted_classes]  # 使用映射將 class 替換為名稱
instances = [item[1] for item in sorted_classes]

plt.figure(figsize=(10, 6))
plt.bar(categories, instances, color='skyblue')
plt.xlabel("Defect Types")
plt.ylabel("Instances")
plt.title("Class Distribution in Dataset")
plt.xticks(rotation=45, ha='right')  # 水平對齊設為右對齊
plt.tight_layout()
plt.show(block=True)

# 列印每個類別的統計數據
print("Class Distribution:")
for class_label, count in sorted_classes:
    print(f"{class_mapping[class_label]}: {count} instances")