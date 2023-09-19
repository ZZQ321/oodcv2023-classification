import os
import pdb
# 数据集文件夹路径
dataset_name = 'robin'
dataset_folder = "/home/zzq/data/OOD-CV-cls-2023/train"

# 获取所有域文件夹的名称
domain_folders = [domain for domain in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, domain))]

# 遍历每个域
for domain in domain_folders:
    domain_path = os.path.join(dataset_folder, domain)
    
    # 获取该域内所有类文件夹的名称
    class_folders = [cls for cls in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, cls))]
    
    # 创建一个保存路径和类别的txt文件
    txt_filename = f"{domain}.txt"
    txt_path = './dataset/' + dataset_name + '/' + txt_filename
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "w") as txt_file:
        # 遍历每个类
        for class_idx, class_folder in enumerate(class_folders):
            class_path = os.path.join(domain_path, class_folder)
            
            # 获取该类内所有图片文件的路径
            image_files = [os.path.join(class_path, img) for img in os.listdir(class_path) ] #if img.endswith(".jpg")
            
            # 将图片路径和对应的类别写入txt文件
            for img_path in image_files:
                txt_file.write(f"{img_path} {class_idx}\n")

print("生成txt文件完成。")
