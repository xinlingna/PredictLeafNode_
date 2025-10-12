#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将质心文件从TXT格式转换为NPZ格式

TXT格式要求：
- 每一行表示一个质心向量
- 每个维度之间的数字用空格隔开
- 支持浮点数和整数

NPZ输出格式：
- 包含一个名为'centroids'的数组
- 形状为[K, D]，其中K是质心数量，D是向量维度
- 数据类型为float32

使用示例：
python convert_centriods_txt_to_npz.py input.txt output.npz
python convert_centriods_txt_to_npz.py --input centroids.txt --output centroids.npz --array_name centroids
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np


def read_centroids_txt(file_path: str, max_centroids: Optional[int] = None) -> np.ndarray:
    """
    读取TXT格式的质心文件
    
    Args:
        file_path: TXT文件路径
        max_centroids: 最大读取的质心数量（用于测试或内存限制）,None 表示不限制
    
    Returns:
        centroids: [K, D] 形状的numpy数组，K为质心数量，D为向量维度
    """
    print(f"正在读取质心文件: {file_path}")
    
    centroids = []
    vector_dim = None
    line_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                
                # 跳过注释行（以#开头）
                if line.startswith('#'):
                    continue
                
                try:
                    # 按空格分割并转换为浮点数
                    values = [float(x) for x in line.split()]
                    
                    if not values:
                        print(f"警告: 第{line_num}行为空，跳过")
                        continue
                    
                    # 检查维度一致性
                    if vector_dim is None:
                        vector_dim = len(values)
                        print(f"检测到向量维度: {vector_dim}")
                    elif len(values) != vector_dim:
                        raise ValueError(f"第{line_num}行维度不匹配: 期望{vector_dim}，实际{len(values)}")
                    
                    centroids.append(values)
                    line_count += 1
                    
                    # 显示进度
                    if line_count % 1000 == 0:
                        print(f"  已读取 {line_count} 个质心")
                    
                    # 检查最大数量限制
                    if max_centroids is not None and line_count >= max_centroids:
                        print(f"达到最大质心数量限制: {max_centroids}")
                        break
                        
                except ValueError as e:
                    print(f"错误: 第{line_num}行解析失败: {line}")
                    print(f"  原因: {str(e)}")
                    raise
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {file_path}")
    except IOError as e:
        raise IOError(f"读取文件失败: {file_path}, 错误: {str(e)}")
    
    if not centroids:
        raise ValueError(f"没有读取到有效的质心向量: {file_path}")
    
    # 转换为numpy数组
    centroids_array = np.array(centroids, dtype=np.float32)
    print(f"读取完成: {centroids_array.shape[0]} 个质心，维度 {centroids_array.shape[1]}")
    
    return centroids_array


def save_centroids_npz(
    centroids: np.ndarray, 
    output_path: str, 
    array_name: str = 'centroids',
    verify: bool = True
) -> None:
    """
    保存质心到NPZ文件
    
    Args:
        centroids: 质心数组 [K, D]
        output_path: 输出NPZ文件路径
        array_name: NPZ中数组的键名
        verify: 是否验证保存的文件
    """
    print(f"正在保存质心到: {output_path}")
    print(f"  数组名称: {array_name}")
    print(f"  形状: {centroids.shape}")
    print(f"  数据类型: {centroids.dtype}")
    
    try:
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存NPZ文件
        np.savez_compressed(output_path, **{array_name: centroids})
        print(f"保存成功: {output_path}")
        
        # 验证保存的文件
        if verify:
            print("正在验证保存的文件...")
            try:
                loaded_data = np.load(output_path)
                if array_name not in loaded_data:
                    raise ValueError(f"保存的文件中没有找到数组 '{array_name}'")
                
                loaded_centroids = loaded_data[array_name]
                if not np.allclose(centroids, loaded_centroids):
                    raise ValueError("保存的数据与原始数据不匹配")
                
                print(f"验证成功: NPZ文件包含键 ['{array_name}']，形状 {loaded_centroids.shape}")
                
                # 显示统计信息
                print(f"质心统计信息:")
                print(f"  最小值: {loaded_centroids.min():.6f}")
                print(f"  最大值: {loaded_centroids.max():.6f}")
                print(f"  均值: {loaded_centroids.mean():.6f}")
                print(f"  标准差: {loaded_centroids.std():.6f}")
                
            except Exception as e:
                print(f"验证失败: {str(e)}")
                raise
        
    except Exception as e:
        print(f"保存失败: {str(e)}")
        raise


def convert_centroids_txt_to_npz(
    input_path: str,
    output_path: str,
    array_name: str = 'centroids',
    max_centroids: Optional[int] = None,
    verify: bool = True
) -> None:
    """
    将TXT格式的质心文件转换为NPZ格式
    
    Args:
        input_path: 输入TXT文件路径
        output_path: 输出NPZ文件路径
        array_name: NPZ中数组的键名
        max_centroids: 最大读取的质心数量
        verify: 是否验证转换结果
    """
    print("=" * 60)
    print("质心文件格式转换: TXT → NPZ")
    print("=" * 60)
    
    try:
        # 检查输入文件
        if not Path(input_path).exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        file_size = Path(input_path).stat().st_size
        print(f"输入文件: {input_path}")
        print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
        
        # 读取TXT文件
        centroids = read_centroids_txt(input_path, max_centroids)
        
        # 保存NPZ文件
        save_centroids_npz(centroids, output_path, array_name, verify)
        
        print("=" * 60)
        print("转换完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
     description="将质心文件从TXT格式转换为NPZ格式",
    )
    # 位置参数
    parser.add_argument('input', nargs='?', help='输入TXT文件路径')
    parser.add_argument('output', nargs='?', help='输出NPZ文件路径')
    
    # 可选参数
    parser.add_argument('-i', '--input', dest='input_file', help='输入TXT文件路径')
    parser.add_argument('-o', '--output', dest='output_file', help='输出NPZ文件路径')
    parser.add_argument('--array_name', default='centroids', help='NPZ中数组的键名 (默认: centroids)')
    parser.add_argument('--max_centroids', type=int, help='最大读取的质心数量')
    parser.add_argument('--no-verify', action='store_true', help='不验证转换结果')
    
    args = parser.parse_args()
    
    # 确定输入输出文件路径
    input_path = args.input or args.input_file
    output_path = args.output or args.output_file
    
    if not input_path:
        parser.error("必须提供输入文件路径")
    if not output_path:
        parser.error("必须提供输出文件路径")
    
    return input_path, output_path, args.array_name, args.max_centroids, not args.no_verify


def main():
    """主函数"""
    try:
        input_path, output_path, array_name, max_centroids, verify = parse_args()
        
        convert_centroids_txt_to_npz(
            input_path=input_path,
            output_path=output_path,
            array_name=array_name,
            max_centroids=max_centroids,
            verify=verify
        )
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n意外错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


'''
cd /home/xln/PycharmProjects/PredictLeafNode/
python src/model/dynamic_k/convert_centriods_txt_to_npz.py \
    --input input/Training_data/siftsmall/siftsmall5H/leaf_center.txt \
    --output input/Training_data/siftsmall/siftsmall5H/leaf_center.npz
'''