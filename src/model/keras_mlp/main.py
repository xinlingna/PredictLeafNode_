from learning import Learning
from testing import Testing
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent

train_file = base_dir / 'input/Training_data/gist1M/leafsize5000/leaf_centroids.txt'
print("train_file:", train_file)
print("train_file exists:", train_file.exists())


def main():
    # 设置文件路径
    train_file = base_dir / 'input/train.txt'
    train_label_file = base_dir / 'input/train_labels.txt'
    test_file = base_dir / 'input/test.txt'
    test_label_file = base_dir / 'input/test_labels.npz'

    # 训练模型
    print("=== 开始训练模型 ===")
    Learning(train_file, train_label_file)

    # 测试模型
    print("=== 开始测试模型 ===")
    Testing(test_file, test_label_file)

# if __name__ == '__main__':
#     main()
