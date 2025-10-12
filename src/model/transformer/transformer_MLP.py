import torch
import numpy as np
from transformers import BertModel, BertConfig, ViTModel, ViTConfig
from typing import Union, Optional, List
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from torch import nn
class CustomTransformer(nn.Module):
    def __init__(self, input_dim=128, hiddle_dim=256,num_layers=3):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=4, dim_feedforward=512
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        return self.transformer(x)  # 输入形状: (seq_len, batch, feature_dim)
    
class TransformerFeatureExtractor:
    """封装多种Transformer方案处理SIFT特征"""
    
    def __init__(self, model_type: str = "bert", feature_dim: int = 128, device: str = "cuda"):
        """
        初始化
        :param model_type: "bert" | "vit" | "custom"
        :param feature_dim: SIFT特征维度（默认128）
        :param device: "cpu" 或 "cuda"
        """
        self.model_type = model_type
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.model = self._load_model()
        
    def _load_model(self):
        """加载预训练模型或自定义模型"""
        if self.model_type == "vit":
            # 方案2：ViT处理图像化SIFT（假设reshape为16x8）
            config = ViTConfig(
                image_size=16,
                patch_size=4,
                num_channels=1,
                hidden_size=self.feature_dim
            )
            model = ViTModel(config)
        elif self.model_type == "custom":
            # 方案3：自定义轻量Transformer（需实现模型类）
            model = self._build_custom_transformer()
        else:
            raise ValueError("model_type must be 'bert', 'vit', or 'custom'")
        return model.to(self.device)
    
    def _build_custom_transformer(self):
        
        return CustomTransformer(input_dim=self.feature_dim)
    
    def extract_features(
        self, 
        sift_features: Union[np.ndarray, torch.Tensor], 
        max_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        提取高层特征
        :param sift_features: 输入SIFT特征，形状为:
            - BERT: (batch_size, num_descriptors, 128)
            - ViT: (batch_size, 128) -> 自动reshape为 (batch_size, 1, 16, 8)
        :param max_seq_len: 截断/填充的序列长度（仅对BERT有效）
        :return: 特征张量，形状取决于模型类型
        """
        if isinstance(sift_features, np.ndarray):
            sift_features = torch.from_numpy(sift_features).float()
        sift_features = sift_features.to(self.device)
        
        if self.model_type == "bert":
            # 处理序列输入
            if max_seq_len:
                sift_features = sift_features[:, :max_seq_len, :]
            outputs = self.model(inputs_embeds=sift_features)
            return outputs.last_hidden_state.mean(dim=1)  # 池化：取序列均值
            
        elif self.model_type == "vit":
            # 处理图像输入（假设原始SIFT是128维，reshape为16x8）
            batch_size = sift_features.shape[0]
            images = sift_features.view(batch_size, 1, 16, 8)  # reshape
            outputs = self.model(pixel_values=images)
            return outputs.last_hidden_state[:, 0, :]  # 取[CLS] token
            
        elif self.model_type == "custom":
            # 自定义Transformer需转置输入为 (seq_len, batch, dim)
            outputs = self.model(sift_features.transpose(0, 1))
            return outputs.mean(dim=0)  # 池化
    

    def save_model(self, save_path: str):
        """保存模型权重"""
        torch.save(self.model.state_dict(), save_path)
        
    def load_model(self, load_path: str):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        
        
class TransformerClassifier(nn.Module):
    """支持特征提取 + MLP分类头联合训练的封装类"""
    
    def __init__(
        self,
        model_type: str = "bert",
        feature_dim: int = 128,
        num_classes: int = 187,
        hidden_dims: List[int] = [256, 128],
        device: str = "cpu",
        freeze_backbone: bool = False
    ):
        """
        :param model_type: "bert" | "vit" | "custom"
        :param feature_dim: SIFT特征维度
        :param num_classes: 分类类别数（如187）
        :param hidden_dims: MLP隐藏层维度列表
        :param device: "cpu" 或 "cuda"
        :param freeze_backbone: 是否冻结Transformer权重（仅训练MLP）
        """
        super().__init__()
        self.device = torch.device(device)
        
        # 1. 初始化特征提取器
        self.feature_extractor = TransformerFeatureExtractor(
            model_type=model_type,
            feature_dim=feature_dim,
            device=device
        )
        
        # 2. 添加MLP分类头
        mlp_layers = []
        input_dim = feature_dim  # Transformer输出维度与输入相同（见之前的实现）
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        mlp_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*mlp_layers).to(self.device)
        
        # 3. 冻结Transformer权重（可选）
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
    def forward(self, sift_features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """前向传播：特征提取 + 分类"""
        features = self.feature_extractor.extract_features(sift_features)
        return self.classifier(features)
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        optimizer: Optimizer = None,
        criterion: nn.Module = None,
        save_path: Optional[str] = None
    ) -> dict:
        """
        训练模型
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器（可选）
        :param epochs: 训练轮次
        :param optimizer: 优化器（默认Adam）
        :param criterion: 损失函数（默认CrossEntropyLoss）
        :param save_path: 模型保存路径（可选）
        :return: 训练日志字典
        """
        self.train()
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=1e-4)
        criterion = criterion or nn.CrossEntropyLoss()
        
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 记录训练损失
            history["train_loss"].append(epoch_loss / len(train_loader))
            
            # 验证集评估
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {history['train_loss'][-1]:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
            
            # 保存最佳模型
            if save_path and val_loader and (val_acc == max(history["val_acc"])):
                torch.save(self.state_dict(), save_path)
        
        return history
    
    def evaluate(self, data_loader: DataLoader, criterion: nn.Module = None) -> tuple:
        """评估模型"""
        self.eval()
        criterion = criterion or nn.CrossEntropyLoss()
        total_loss, correct = 0.0, 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                total_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
        
        return (
            total_loss / len(data_loader),
            correct / len(data_loader.dataset)
        )

# 辅助函数：创建数据加载器
def create_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """从NumPy数组创建DataLoader"""
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).long()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
