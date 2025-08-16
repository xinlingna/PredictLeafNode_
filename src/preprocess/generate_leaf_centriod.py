# 获取每个叶子节点地质心保存在文件中

import os
import numpy as np

# 指定叶子节点文件目录路径
directory = '../input/Training_data'
# 指定质心输出文件
output_centriod_file= '../../output/output_centriod_file.txt'

centriods_List=[]
# 遍历目录中的所有文件
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # 检查是否为文件（跳过目录）
    if os.path.isfile(file_path):
        try:
            # 加载数据
            data = np.loadtxt(file_path)

            # 计算均值质心
            centroid = np.mean(data, axis=0)
            centriods_List.extend([centroid])
        except Exception as e:
            print('加载数据失败')

if centriods_List:
    centroids_array = np.array(centriods_List)
    np.savetxt(output_centriod_file, centroids_array, fmt='%.6f')

print(f"已保存 {len(centriods_List)} 个质心到文件：{output_centriod_file}")


# # 测试写入数据是否正确
# loadtxt = np.loadtxt(output_centriod_file)
# print(loadtxt.shape) # (2, 10)
