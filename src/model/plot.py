from typing import List, Dict
import matplotlib.pyplot as plt
import os

def plot_epoch1_batch_metrics(
    epoch1_batch_metrics: List[Dict[str, float]],
    save_dir: str,
    save_name: str,
    eval_topk: int,
):
    if len(epoch1_batch_metrics) == 0:
        return
    if not save_dir or not os.path.isdir(save_dir):
        print("Warning: Invalid save directory, skipping epoch1 batch metrics plotting")
        return
    
    # Check if all required metrics are present
    required_metrics = ['batch', 'val_ordacc', 'val_recall', 'val_kld', 'train_ordacc', 'train_recall', 'train_kld']
    if not all(all(metric in m for metric in required_metrics) for m in epoch1_batch_metrics):
        print("Warning: Some required metrics are missing in epoch1_batch_metrics")
        return
        
    batches = [m['batch'] for m in epoch1_batch_metrics]
    val_ordacc_values = [m['val_ordacc'] for m in epoch1_batch_metrics]
    val_recall_values = [m['val_recall'] for m in epoch1_batch_metrics]
    val_kld_values = [m['val_kld'] for m in epoch1_batch_metrics]

    train_ordacc_values = [m['train_ordacc'] for m in epoch1_batch_metrics]
    train_recall_values = [m['train_recall'] for m in epoch1_batch_metrics]
    train_kld_values = [m['train_kld'] for m in epoch1_batch_metrics]        
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上图：准确率和召回率
    ax1.plot(batches, val_ordacc_values, marker='o', label=f'val_ord_acc@{eval_topk}', linewidth=2, markersize=4)
    ax1.plot(batches, val_recall_values, marker='s', label=f'val_recall@{eval_topk}', linewidth=2, markersize=4)
    ax1.plot(batches, train_ordacc_values, marker='m', label=f'train_ord_acc@{eval_topk}', linewidth=2, markersize=4, linestyle='--')
    ax1.plot(batches, train_recall_values, marker='n', label=f'train_recall@{eval_topk}', linewidth=2, markersize=4, linestyle='--')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Epoch 1: Validation Cluster_Accuracy and Recall vs Batch')
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend()
    
    # 下图：KL散度损失
    ax2.plot(batches, val_kld_values, marker='^', color='red', label='val_kld', linewidth=2, markersize=4)
    ax2.plot(batches, train_kld_values, marker='^', color='blue', label='train_kld', linewidth=2, markersize=4, linestyle='--')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('KL Divergence Loss')
    ax2.set_title(f'Training & Validation KL Loss vs Batch')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    batch_plot_path = os.path.join(save_dir, os.path.splitext(save_name)[0] + f"_epoch1_batch_metrics@{eval_topk}.png")
    plt.savefig(batch_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved epoch 1 batch metrics plot to {batch_plot_path}")


def plot_epoch_metrics(
    save_dir: str,
    save_name: str,
    epoch_history: List[Dict[str, float]],
):
    if len(epoch_history) == 0:
        return
    if not save_dir or not os.path.isdir(save_dir):
        print("Warning: Invalid save directory, skipping epoch metrics plotting")
        return
    
    # Check if all required metrics are present
    required_metrics = ['val_ordacc', 'val_recall', 'val_kld', 'train_ordacc', 'train_recall', 'train_loss']
    if not all(all(metric in epoch for metric in required_metrics) for epoch in epoch_history):
        print("Warning: Some required metrics are missing in epoch_history")
        return
        
    epochs = list(range(1, len(epoch_history) + 1))
    val_ordacc_values = [epoch['val_ordacc'] for epoch in epoch_history]
    val_recall_values = [epoch['val_recall'] for epoch in epoch_history]
    val_kld_values = [epoch['val_kld'] for epoch in epoch_history]

    train_ordacc_values = [epoch['train_ordacc'] for epoch in epoch_history]
    train_recall_values = [epoch['train_recall'] for epoch in epoch_history]
    train_loss_values = [epoch['train_loss'] for epoch in epoch_history]        
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上图：准确率和召回率
    ax1.plot(epochs, val_ordacc_values, marker='o', label='val_ordacc', linewidth=2, markersize=4)
    ax1.plot(epochs, val_recall_values, marker='s', label='val_recall', linewidth=2, markersize=4)
    ax1.plot(epochs, train_ordacc_values, marker='m', label='train_ordacc', linewidth=2, markersize=4, linestyle='--')
    ax1.plot(epochs, train_recall_values, marker='n', label='train_recall', linewidth=2, markersize=4, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Validation and Training Accuracy & Recall vs Epoch')
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend()
    
    # 下图：损失
    ax2.plot(epochs, val_kld_values, marker='^', color='red', label='val_kld', linewidth=2, markersize=4)
    ax2.plot(epochs, train_loss_values, marker='^', color='blue', label='train_loss', linewidth=2, markersize=4, linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss & Validation KL Loss vs Epoch')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    epoch_plot_path = os.path.join(save_dir, os.path.splitext(save_name)[0] + "_epoch_metrics.png")
    plt.savefig(epoch_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved epoch metrics plot to {epoch_plot_path}")
    
    # Print best validation KLD
    best_val_kld = min(val_kld_values)
    best_epoch_kld = val_kld_values.index(best_val_kld) + 1
    print(f"best val_kld: {best_val_kld:.6f} at epoch {best_epoch_kld}")