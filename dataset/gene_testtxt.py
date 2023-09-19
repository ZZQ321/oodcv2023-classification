import os
import pdb
# 数据集文件夹路径
dataset_name = 'robin'
dataset_folder = "/home/zzq/data/OOD-CV-cls-2023/"

# 获取所有域文件夹的名称
domain_folders = [domain for domain in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, domain))]

# 遍历每个域
dataset_name='robin'
txt_filename= 'phase2-cls.txt'
txt_path = './dataset/' + dataset_name + '/' + txt_filename
os.makedirs(os.path.dirname(txt_path), exist_ok=True)

with open(txt_path, "w") as txt_file:

    class_path = '/home/zzq/data/OOD-CV-cls-2023/phase2-cls/images'
    
    # 获取该类内所有图片文件的路径
    image_files = [os.path.join(class_path, img) for img in os.listdir(class_path) ] #if img.endswith(".jpg")
    
    # 将图片路径和对应的类别写入txt文件
    for img_path in image_files:
        txt_file.write(f"{img_path} 0\n")

print("生成txt文件完成。")
