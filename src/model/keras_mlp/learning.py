import numpy as np
import os
from time import time, strftime, localtime
from mlp import build_model
from keras.models import save_model
from keras.callbacks import EarlyStopping
import argparse
from filelock import FileLock
import datetime
import matplotlib.pyplot as plt


def plot_training_metrics(history_dict, output_dir, top_k=10, model_name=""):
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 设置曲线颜色和样式
    train_color = '#1f77b4'  # 蓝色
    val_color = '#ff7f0e'    # 橙色
    
    # 获取数据
    train_top_k = history_dict.get('lambda', [])
    val_top_k = history_dict.get('val_lambda', [])
    epochs = range(1, len(train_top_k) + 1) if train_top_k else range(1, len(val_top_k) + 1)
    
    # 绘制训练集曲线
    if train_top_k:
        plt.plot(epochs, train_top_k, 
                color=train_color, 
                linestyle='-', 
                linewidth=2,
                label=f'Train Top-{top_k} Accuracy')
        
        # 标记训练集最高点
        max_train = np.max(train_top_k)
        max_train_epoch = np.argmax(train_top_k) + 1
        plt.scatter(max_train_epoch, max_train, 
                   color=train_color, 
                   s=100, 
                   alpha=0.7,
                   edgecolors='black')
        plt.annotate(f'Max: {max_train:.4f}', 
                    xy=(max_train_epoch, max_train),
                    xytext=(10, 10),
                    textcoords='offset points',
                    ha='left',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc=train_color, alpha=0.2))
    
    # 绘制验证集曲线
    if val_top_k:
        plt.plot(epochs, val_top_k, 
                color=val_color, 
                linestyle='--', 
                linewidth=2,
                label=f'Validation Top-{top_k} Accuracy')
        
        # 标记验证集最高点
        max_val = np.max(val_top_k)
        max_val_epoch = np.argmax(val_top_k) + 1
        plt.scatter(max_val_epoch, max_val, 
                   color=val_color, 
                   s=100, 
                   alpha=0.7,
                   edgecolors='black')
        plt.annotate(f'Max: {max_val:.4f}', 
                    xy=(max_val_epoch, max_val),
                    xytext=(10, -15),
                    textcoords='offset points',
                    ha='left',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc=val_color, alpha=0.2))
    
    # 图形装饰
    title = f'Top-{top_k} Accuracy vs. Epochs'
    if model_name:
        title = f'{model_name}\n{title}'
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(f'Top-{top_k} Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 调整坐标轴范围
    if train_top_k or val_top_k:
        all_values = train_top_k + val_top_k
        plt.ylim(min(all_values)*0.95, min(max(all_values)*1.05, 1.0))
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"topk_acc_comparison_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

# === log函数：写入日志文件（由主函数传入打开好的句柄）===
def log(message, file_handle):
    print(message)
    file_handle.write(message + '\n')

# === 主训练函数 ===
def Learning(input_file, 
             model_file_ = None,      
             hidden_units = 1024,
             batch_size = 256,
             epochs = 100,
             top_k=10):
    np.random.seed(1337) 
    
    # 日志文件路径（独立，每次不同）
    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    pid = os.getpid()
    params_str = f"hu{hidden_units}_bs{batch_size}_ep{epochs}_tk{top_k}"
    log_path = os.path.join(
        os.path.dirname(input_file),
        f"model_log_{timestamp}_pid{pid}_{params_str}.txt"
    )
    
    filename_no_ext = os.path.splitext(os.path.basename(input_file))[0]
    model_file=os.path.join(os.path.dirname(input_file), f"model_{filename_no_ext}.h5")
    if(model_file_!=None):
        model_file=model_file_

    # 只打开一次文件，传递句柄
    with open(log_path, 'a', encoding='utf-8') as log_file:
           log("=" * 60, log_file)
           current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           log(f"\n[{current_time}]", log_file)

           # 加载数据
           data = np.load(input_file)
           input_query = data['query']
           input_sim=data['sim']
           y_train = data['label']
           print("y_train.shape:",y_train.shape)
           print("input_quey.shape:",input_query.shape)
           print("input_sim.shape:",input_sim.shape)
           assert np.allclose(np.sum(y_train, axis=1), 1.0, atol=1e-6), "Soft labels must sum to 1"
           output_units = y_train.shape[1]

           start_time = time()

           log(f"Model Parameters:", log_file)
           log(f"input_files:{os.path.basename(input_file)}", log_file)
           log(f" - Hidden Units: {hidden_units}", log_file)
           log(f" - Batch Size:   {batch_size}", log_file)
           log(f" - Epochs:       {epochs}", log_file)
           

           # 构建模型
           model = build_model(input_query_dim=input_query.shape[1], input_sim_dim=input_sim.shape[1],output_dim=output_units, hidden_units=hidden_units,top_k=top_k)
           # 输出模型结构到日志
           log(f"\nModel Architecture:\n", log_file)
           model.summary(print_fn=lambda x: log(x, log_file))

           # 训练模型
           early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
           history = model.fit(
                               x={'query_input': input_query, 'sim_input': input_sim},
                               y=y_train,
                               validation_split=0.2,
                               epochs=epochs,
                               batch_size=batch_size,
                            #    callbacks=[early_stop],
                               verbose=1) # 	每个 epoch 输出一行进度条（默认）

           # 保存模型
           model.save(model_file,os.path.dirname(input_file),)

           # 输出耗时
           end_time = time()
           time_taken = end_time - start_time
           hours, rest = divmod(time_taken, 3600)
           minutes, seconds = divmod(rest, 60)
           log(f"Training results:", log_file)
           log(f"- Training Time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds", log_file)


           # 训练统计
           history_dict = history.history
           actual_epochs = len(history_dict['loss'])
           log(f"- Actual Epochs Trained: {actual_epochs}", log_file)
           
           # 绘制图像
           plot_training_metrics(history_dict,os.path.dirname(input_file),top_k)

           acc_key = 'accuracy' if 'accuracy' in history_dict else 'categorical_accuracy'
           train_acc = history_dict[acc_key]
           train_loss = history_dict['loss']

           max_acc_epoch = np.argmax(train_acc) + 1
           min_loss_epoch = np.argmin(train_loss) + 1


           if 'val_accuracy' in history_dict or 'val_categorical_accuracy' in history_dict:
               val_acc_key = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_categorical_accuracy'
               val_acc = history_dict[val_acc_key]
               val_loss = history_dict['val_loss']

               max_val_acc_epoch = np.argmax(val_acc) + 1
               min_val_loss_epoch = np.argmin(val_loss) + 1

               log(f"-  Max Train Accuracy: {train_acc[max_acc_epoch-1]:.4f} at epoch {max_acc_epoch}", log_file)
               log(f"- Max Val Accuracy:   {val_acc[max_val_acc_epoch-1]:.4f} at epoch {max_val_acc_epoch}", log_file)
               log(f"- Min Train Loss:     {train_loss[min_loss_epoch-1]:.4f} at epoch {min_loss_epoch}", log_file)
               log(f"- Min Val Loss:       {val_loss[min_val_loss_epoch-1]:.4f} at epoch {min_val_loss_epoch}", log_file)
            
           if 'lambda' in history_dict:
               tk=history_dict['lambda']
               max_tk=np.argmax(tk) + 1
               log(f"- Max Top-K accuracy: , {tk[max_tk-1]:.4f} at epoch {max_tk}", log_file)
               
           if 'val_lambda' in history_dict:
               tk=history_dict['val_lambda']
               max_tk=np.argmax(tk)+1
               log(f"- Max val Top-K accuracy: , {tk[max_tk-1]:.4f} at epoch {max_tk}", log_file)
                
               
           # KL散度
           if 'kullback_leibler_divergence' in history_dict:
               kl_train = history_dict['kullback_leibler_divergence']
               min_kl_epoch = np.argmin(kl_train) + 1
               log(f"- Min Train KL Divergence: {kl_train[min_kl_epoch-1]:.4f} at epoch {min_kl_epoch}", log_file)

           if 'val_kullback_leibler_divergence' in history_dict:
               kl_val = history_dict['val_kullback_leibler_divergence']
               min_val_kl_epoch = np.argmin(kl_val) + 1
               log(f"- Min Val KL Divergence:   {kl_val[min_val_kl_epoch-1]:.4f} at epoch {min_val_kl_epoch}", log_file)
            
            
            
""" # 示例调用
if __name__ == "__main__":
    input_file='input/Training_data/sift1M/leafsize10000/output_training_file.npz'
    model_file="input/Training_data/sift1M/leafsize10000/model.h5"
    Learning(input_file, model_file) """
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a neural network with optional hyperparameters.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input .npz data file')
    parser.add_argument('--model_file', type=str, default=None, help='Path to save the trained model')

    parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--top_k', type=int, default=20, help='probability of top-k leaf nodes')

    args = parser.parse_args()

    Learning(
        input_file=args.input_file,
        model_file_=args.model_file,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        epochs=args.epochs,
        top_k=args.top_k
    )

