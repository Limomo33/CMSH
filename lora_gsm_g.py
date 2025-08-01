# 在模型定义部分新增以下内容
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import log
import os
from torch.utils.data import DataLoader, TensorDataset,Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import esm
from foutput import *
import torch.nn.functional as F
import math
from adj import *
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from typing import List, Optional

import json
import gc
# from esm.modules import LoRA
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # 冻结原始权重
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # 创建 LoRA 适配器
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化 LoRA 权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        base_output = self.base_layer(x)

        # LoRA 部分
        lora_output = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        lora_output = lora_output * self.scaling

        return base_output + lora_output


class ESM2WithLoRA(nn.Module):
    def __init__(self, esm_model, target_modules: Optional[List[str]] = None,
                 r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.model = esm_model

        # 默认目标模块
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj"]
        self.modified_layers = []
        self._apply_lora(target_modules, r, lora_alpha, lora_dropout)

        # 记录已修改的模块


    def _apply_lora(self, target_modules: List[str], r: int, lora_alpha: int, lora_dropout: float):
        """应用 LoRA 适配器到目标模块"""
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # 检查是否为目标模块
            is_target = any(target in name for target in target_modules)
            if is_target:
                parent = self._get_parent_module(name)
                child_name = name.split('.')[-1]

                # 替换为 LoRALinear
                lora_layer = LoRALinear(module, r, lora_alpha, lora_dropout)
                setattr(parent, child_name, lora_layer)
                self.modified_layers.append(name)

    def _get_parent_module(self, module_path: str):
        """获取父模块对象"""
        parts = module_path.split('.')
        current = self.model
        for part in parts[:-1]:  # 排除最后一部分
            current = getattr(current, part)
        return current

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"可训练参数: {trainable_params} / 总参数: {total_params}")
        print(f"可训练参数占比: {100 * trainable_params / total_params:.2f}%")
        print(f"已应用 LoRA 的层: {self.modified_layers}")

class ESMModelWrapper(nn.Module):
    def __init__(self, esm_model_name="esm2_t6_8M_UR50D"):
        super().__init__()
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(esm_model_name)
        # self.batch_converter = self.alphabet.get_batch_converter()
        self.contrast_proj = nn.Sequential(
            nn.Linear(self.esm_model.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 冻结ESM参数
        for param in self.esm_model.parameters():
            param.requires_grad_(False)

        # 获取输出维度
        self.repr_layer = self.esm_model.num_layers
        self.embed_dim = self.esm_model.embed_dim

    def forward(self, sequences):
        # sequences = sequences.squeeze().to(device)
        # print(sequences.size())
        # batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        with torch.no_grad():
            results = self.esm_model(sequences, repr_layers=[self.repr_layer])
        token_embeddings = results["representations"][self.repr_layer]
        # print(token_embeddings.size())
        contrast_emb = self.contrast_proj(token_embeddings.mean(1))

        # 去除CLS和SEP token
        # embeddings = self._remove_special_tokens(token_embeddings, sequences)
        return token_embeddings, contrast_emb


class EnhancedESMModelWrapper(nn.Module):
    def __init__(self, esm_model_name: str = "esm2_t6_8M_UR50D",
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(esm_model_name)

        # 应用 LoRA
        self.esm_model = ESM2WithLoRA(
            esm_model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # 打印可训练参数信息
        self.esm_model.print_trainable_parameters()

        # 对比学习投影头
        self.contrast_proj = nn.Sequential(
            nn.Linear(esm_model.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, sequences):
        # 前向传播时同时返回原始嵌入和对比嵌入
        # from peft import inject_adapter_in_model
        # self.esm_model = inject_adapter_in_model(lora_config, self.esm_model)
        results = self.esm_model(sequences, repr_layers=[self.esm_model.model.num_layers])
        # print(results)
        token_embeddings = results["representations"][self.esm_model.model.num_layers]
        contrast_emb = self.contrast_proj(token_embeddings.mean(1))
        lables = results["logits"]
        # print('111111111111111',lables.size(),token_embeddings.size())
        return token_embeddings, contrast_emb, lables


def compute_loss(logits, labels):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

# class GraphAttentionLayer2(nn.Module):
#     def __init__(self, in_features, out_features, batch_size=256, num_head=2, dropout=0.5, alpha=0.2, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#         self.num_head = num_head
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#
#         self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         # self.Adj = nn.Parameter(torch.ones(size=(batch_size, 300, 300)))
#         # nn.init.xavier_uniform_(self.Adj.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, x, adj):
#         # print(x.size())
#         # print(self.W.size())
#
#         batch_size, num_nodes, in_feat = x.size()
#         h = torch.matmul(x, self.W)
#
#         # N = x.size()[0]
#         # h_flat = h.view(batch_size, -1)
#         # print('111111111111', adj.size())
#         # a_input = torch.cat([h.repeat(1, 1,N).view(N * N, self.out_features), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
#
#         h = F.dropout(h, self.dropout, training=self.training)
#         Wh1 = torch.matmul(h, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(h, self.a[self.out_features:, :])
#         e = Wh1 + Wh2.permute(0, 2, 1)
#         # print('111111111111', Wh1.size(),Wh2.size())
#
#         # a_input = torch.cat([h.repeat(1, seq_len).view(N, seq_len * seq_len, -1),
#         #                      h.repeat(1, seq_len, 1)], dim=-1)
#         # a_input = a_input.view(N, seq_len, seq_len, 2 * self.out_features)
#
#         attention = self.leakyrelu(e)
#         # print('111111', attention.size())
#
#         # zero_vec = -10e9 * torch.ones_like(attention)
#         # mask=zero_vec* self.Adj
#         # attention = torch.where(adj > 0, attention, zwero_vec)
#         attention = attention * adj#[:attention.size()[0], :, :]
#         # print('22222222', attention.size())
#
#         attention = F.softmax(attention, dim=1)
#
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, h)
#         # print('111111111111', h_prime.size())
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, batch_size=256, num_head=2, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_head = num_head
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # self.Adj = nn.Parameter(torch.ones(size=(batch_size, 300, 300)))
        # nn.init.xavier_uniform_(self.Adj.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        # print(x.size())
        # print(self.W.size())

        batch_size, num_nodes, in_feat = x.size()
        h = torch.matmul(x, self.W)

        # N = x.size()[0]
        # h_flat = h.view(batch_size, -1)
        # print('111111111111', adj.size())
        # a_input = torch.cat([h.repeat(1, 1,N).view(N * N, self.out_features), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)

        h = F.dropout(h, self.dropout, training=self.training)
        Wh1 = torch.matmul(h, self.a[:self.out_features, :])
        Wh2 = torch.matmul(h, self.a[self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        # print('111111111111', Wh1.size(),Wh2.size())

        # a_input = torch.cat([h.repeat(1, seq_len).view(N, seq_len * seq_len, -1),
        #                      h.repeat(1, seq_len, 1)], dim=-1)
        # a_input = a_input.view(N, seq_len, seq_len, 2 * self.out_features)

        attention = self.leakyrelu(e)
        # print('111111', attention.size())

        # zero_vec = -10e9 * torch.ones_like(attention)
        # mask=zero_vec* self.Adj
        # attention = torch.where(adj > 0, attention, zwero_vec)
        attention = attention * adj#[:attention.size()[0], :, :]
        # print('22222222', attention.size())

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        # print('111111111111', h_prime.size())

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class BatchGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        """
        批量图卷积网络模型

        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GCN层数
            dropout: Dropout概率
        """
        super(BatchGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # 创建固定的邻接矩阵模板
        adj_template = self.create_adj_template()
        # 转换为稀疏格式并注册为buffer
        edge_index, edge_weight = dense_to_sparse(adj_template)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # 创建GCN层
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = output_dim if i == num_layers - 1 else hidden_dim
            # 注意：添加add_self_loops=False因为模板已包含自环
            self.layers.append(GCNConv(in_channels, out_channels, add_self_loops=False))

        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.init_weights()

    def create_adj_template(self):
        """创建324x324的固定邻接矩阵模板"""
        size = 324
        adj = torch.eye(size, dtype=torch.float)  # 对角线为1

        # 指定的连接索引
        indices = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99,
                   114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]

        # 设置301-324行（索引300-323）的指定列
        for row in range(300, 324):
            for col in indices:
                adj[row, col] = 1
                adj[col, row] = 1  # 对称位置

        return adj

    def init_weights(self):
        """Xavier初始化权重"""
        for layer in self.layers:
            if hasattr(layer, 'lin'):  # GCNConv有lin属性
                nn.init.xavier_uniform_(layer.lin.weight)
                if layer.lin.bias is not None:
                    nn.init.zeros_(layer.lin.bias)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 (batch_size, num_nodes, input_dim)
            adj: 邻接矩阵 (batch_size, num_nodes, num_nodes)

        返回:
            输出张量 (batch_size, num_nodes, output_dim)
        """
        batch_size, num_nodes, _ = x.size()

        # 初始化输出列表
        batch_outputs = []

        # 处理每个样本
        for i in range(batch_size):
            # 获取当前样本的数据
            sample_x = x[i]  # (num_nodes, input_dim)

            # 将密集邻接矩阵转换为PyG需要的稀疏格式

            # 通过GCN层传递数据
            for j, layer in enumerate(self.layers):
                sample_x = layer(sample_x, self.edge_index, self.edge_weight)

                # 如果不是最后一层，应用ReLU和Dropout
                if j < len(self.layers) - 1:
                    sample_x =  F.leaky_relu(sample_x, negative_slope=0.01)
                    sample_x = F.dropout(sample_x, p=self.dropout, training=self.training)

                    # 批归一化
                    sample_x = self.batch_norm(sample_x)

            batch_outputs.append(sample_x)

        # 将输出堆叠回三维张量
        return torch.stack(batch_outputs, dim=0)
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, n_final_out, dropout=0.5, alpha=0.2, nheads=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([

            GraphAttentionLayer(

                in_features=nfeat,  # 输入特征维度

                out_features=nhid,  # 每个头的输出维度

                dropout=dropout,

                alpha=alpha,

                concat=True

            ) for _ in range(nheads)

        ])

        self.out_att = GraphAttentionLayer(nhid * nheads, n_final_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # print('11111111111',x.size())

        return x



class PositionEmbeddingTrain(nn.Module):
    def __init__(self, max_position, dim):
        super().__init__()
        self.pos_embed = nn.Embedding(max_position, dim)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        positions = self.pos_embed(positions)
        return torch.cat((x, positions), dim=2)


def simcse_unsup_loss(mhc_con: torch.Tensor, pep_con: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    # 1. 归一化嵌入向量
    mhc_con = F.normalize(mhc_con, dim=1)
    pep_con = F.normalize(pep_con, dim=1)
    batch_size = mhc_con.size(0)
    labels = torch.arange(batch_size, device=mhc_con.device)
    sim_matrix = mhc_con @ pep_con.T  # [batch_size, batch_size]
    sim_matrix /= temperature

    # 计算两个方向的损失
    loss_mhc_to_peptide = F.cross_entropy(sim_matrix, labels)
    loss_peptide_to_mhc = F.cross_entropy(sim_matrix.T, labels)

    # 对称损失
    loss = (loss_mhc_to_peptide + loss_peptide_to_mhc) / 2

    return loss


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=2, head_size=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_size = head_size

        self.WQ = nn.Linear(dim, num_heads * head_size)
        self.WK = nn.Linear(dim, num_heads * head_size)
        self.WV = nn.Linear(dim, num_heads * head_size)
        self.out = nn.Linear(num_heads * head_size, dim)

    def forward(self, x):
        batch_size = x.size(0)

        Q = self.WQ(x).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = self.WK(x).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = self.WV(x).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, V).permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.head_size)
        return self.out(output)


# class CustomLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(10, 5))

#     def forward(self, x):
#         noise = torch.normal(mean=torch.zeros(tensor.size()), std=1)
#         tensor_noisy = x + noise
#         return tensor_noisy
import torch
import torch.nn as nn


class TrainableGaussianNoise(nn.Module):
    """
    可训练高斯噪声层

    参数:
    - initial_std (float): 初始标准差 (默认: 0.1)
    - per_channel (bool): 是否为每个通道使用独立噪声 (默认: False)
    - trainable (bool): 噪声标准差是否可训练 (默认: True)
    - always_active (bool): 在eval模式下是否保持激活 (默认: False)
    - clamp_output (tuple): 输出值的钳制范围 (默认: None)
    """

    def __init__(self,
                 initial_std=0.1,
                 per_channel=False,
                 trainable=True,
                 always_active=False,
                 clamp_output=None):
        super().__init__()
        self.per_channel = per_channel
        self.trainable = trainable
        self.always_active = always_active
        self.clamp_output = clamp_output

        # 初始化标准差参数
        if trainable:
            if per_channel:
                # 通道级噪声：每个通道独立标准差
                self.log_std = nn.Parameter(torch.zeros(1) + torch.log(torch.tensor(initial_std)))
            else:
                # 全局噪声：单一标准差
                self.log_std = nn.Parameter(torch.zeros(1) + torch.log(torch.tensor(initial_std)))
        else:
            # 固定标准差
            self.register_buffer('log_std', torch.tensor(initial_std).log())

    def forward(self, x):
        # 仅在训练模式或always_active为True时添加噪声
        if self.training or self.always_active:
            # 计算当前标准差 (确保正值)
            std = torch.exp(self.log_std)

            # 根据输入维度调整噪声形状
            if self.per_channel and x.dim() > 2:
                # 通道级噪声：形状 [1, C, 1, ...]
                noise_shape = [1] * x.dim()
                noise_shape[1] = x.size(1)  # 通道维度
                noise = torch.randn(*noise_shape, device=x.device) * std
            else:
                # 全局噪声：与输入相同形状
                noise = torch.randn_like(x) * std

            # 添加噪声
            x = x + noise

            # 可选：钳制输出值范围
            if self.clamp_output is not None:
                x = torch.clamp(x, *self.clamp_output)

        return x

    def current_std(self):
        """获取当前标准差值"""
        return torch.exp(self.log_std).item()

    def extra_repr(self):
        """显示层配置信息"""
        return (f"std={self.current_std():.4f}, per_channel={self.per_channel}, "
                f"trainable={self.trainable}, always_active={self.always_active}")


# 修改TransformerModel类
class TransformerModel(nn.Module):
    def __init__(self, seq1_len, seq2_len, dim=340, kernel_size=32, num_heads=2, gat_heads=8, fc2_size=1):
        super().__init__()
        # 其他初始化保持不变...
        self.pos_embed1 = PositionEmbeddingTrain(seq1_len, 20)
        self.pos_embed2 = PositionEmbeddingTrain(seq2_len, 20)

        self.attention1 = MultiHeadAttention(340)
        self.attention2 = MultiHeadAttention(340)

        self.gat=GAT(nfeat=20, nhid=256, n_final_out=20)
        self.gconv1 = BatchGCN(20,128,32)
        self.linearmhc=nn.Linear(43820, 5120)
        self.linearmhc2=nn.Linear(5120, 1280)

        self.linear0 = nn.Linear(10368, 2560)  # 根据实际维度调整

        self.linear1 = nn.Linear(9440, 1280)  # 根据实际维度调整
        self.linear2 = nn.Linear(3200, 256)  # 根据实际维度调整
        self.linear3 = nn.Linear(87040, 2560)  # 根据实际维度调整
        self.linear4 = nn.Linear(3840, 2560)  # 11136
        self.linear5 = nn.Linear(5120, 2560)
        self.linear6 = nn.Linear(2560, 256)
        # self.linearc = nn.Linear(2560, 300)

        self.linear7 = nn.Linear(256, 1)

        self.conv1 = nn.Conv1d(
            in_channels=300,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.convp = nn.Conv1d(
            in_channels=24,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.flatten = nn.Flatten()  # 正确使用Flatten

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool2= nn.AvgPool1d(kernel_size=1)
        self.dropout = nn.Dropout(0.5)
        # self.linear=nn.Linear(out_features=2)
        self.norm = nn.LayerNorm(340, eps=1e-5, elementwise_affine=True)
        # self.norm2 = nn.LayerNorm(340, eps=1e-5, elementwise_affine=True)

        # 替换原有的ESM编码器
        self.esm_encoder = EnhancedESMModelWrapper()
        self.esm_encoder2 = ESMModelWrapper()
        self.fixed_noise = TrainableGaussianNoise(
            initial_std=0.2,
            trainable=False,
            always_active=True,
            clamp_output=(-3, 3))

        # 新增对比学习相关参数
        # self.contrast_loss = nn.L1Loss(reduction='mean')
        self.contrast_loss=nn.CosineEmbeddingLoss()
        self.contrast_weight = 0.3  # 对比损失权重

        self.com_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, pep, mhc, adj, mhc_seq, pep_seq):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 获取ESM嵌入和对比嵌入
        pep_embeddings, pep_contrast = self.esm_encoder2(pep_seq)
        mhc_embeddings, mhc_contrast, l = self.esm_encoder(mhc_seq)

        # 其他处理保持不变...
        pep_embeddings = torch.cat([pep_embeddings, pep_embeddings], dim=1)  # 填充到12
        # mhc_embeddings = self._pad_sequences(mhc_embeddings, target_len=300)  # 填充到30
        pepesm = self.pos_embed1(pep_embeddings)
        mhcesm = self.pos_embed2(mhc_embeddings)
        pepesm = self.fixed_noise(pepesm)
        mhcesm = self.dropout(mhcesm)

        # print(pep.size(),mhc.size())

        # pep_attn = self.attention1(pepesm)
        # mhc_attn = self.attention2(mhcesm)  # (batch,seq,dim)
        #
        pep_pool = self.pool2(pepesm)
        mhc_pool = self.pool2(mhcesm)
        # # print('11111',mhc_pool.size())
        mhc_pool = self.dropout(mhc_pool)
        mhc_pool = self.norm(mhc_pool)
        pep_pool = self.norm(pep_pool)

        adj = self.fixed_noise(adj)
        Y = self.gat(mhc, adj)
        combined = torch.cat([Y, pep], dim=1)

        # y2=torch.cat([mhc,flat_mhc_0],dim=1)
        Y2=self.gconv1(combined)
        # Y=torch.mul(gat_out,gcn_out)
        # Y = self.pool(gat_out).squeeze(-1)
        Y3=self.flatten(Y2)
        Y_2 = self.linear0(Y3)
        Y_2 = F.relu(Y_2)
        # print(mhc_att.size())
        # Y_cont = self.linearc(Y_2)

        mhc_conv1 = self.conv1(mhcesm)
        mhc_conv = self.conv2(mhc_conv1)
        mhc_conv = self.dropout(mhc_conv)

        pep_conv = self.convp(pepesm)

        flat_pep_0 = self.flatten(pep_pool)
        flat_pep_1 = self.flatten(pep_conv)

        flat_mhc_0 = self.pool(mhc_pool).squeeze(-1)  #
        # print('11111', mhc_att.size())  # (batch,seq,dim)
        flat_mhc_1 = self.flatten(mhc_conv1)#.squeeze(-1)
        # print('22222222', flat_mhc_1.size())  # (batch,seq,dim)

        flat_mhc_2 = self.flatten(mhc_conv)#.squeeze(-1)

        cat_0=torch.cat([flat_mhc_1,flat_mhc_0],dim=1)
        # print('11111', cat_0.size())
        cat_0 = self.linearmhc(cat_0)
        cat_0 = F.relu(cat_0)
        cat_0 = self.dropout(cat_0)
        cat_0 = self.linearmhc2(cat_0)
        cat_0 = F.relu(cat_0)
        cat_0 = self.dropout(cat_0)
        cat_1 = torch.cat([flat_pep_0, cat_0], dim=1)
        # print('222222', cat_1.size())

        cat_2 = torch.cat([flat_pep_1, flat_mhc_2], dim=1)
        # print('111333331', cat_2.size())

        fc1_0 = self.linear1(cat_1)
        fc1_0 = F.relu(fc1_0)
        fc1_0 = self.dropout(fc1_0)

        # fc1_1 = self.linear2(cat_1)
        # fc1_1=F.relu(fc1_1)
        fc1_2 = self.linear3(cat_2)
        fc1_2 = F.relu(fc1_2)
        fc1_2 = self.dropout(fc1_2)

        merge_1 = torch.cat([fc1_0, fc1_2], dim=1)
        # print('4444444', merge_1.size())

        fc2 = self.linear4(merge_1)  # merge_1
        fc2 = F.relu(fc2)

        merge_2=torch.cat([fc2, Y_2], dim=1)
        # fc2=self.norm1(fc2)
        # print('5555555', merge_2.size())

        fc3 = self.linear5(merge_2)  # merge_1
        fc3 = F.relu(fc3)
        fc3 = self.dropout(fc3)

        fc3 = self.linear6(fc3)  # merge_1
        fc3 = F.relu(fc3)

        out = self.linear7(fc3)
        out = F.leaky_relu(out, negative_slope=0.002)

        contrast_target = torch.ones(Y_2.size(0), device=Y_2.device)
        contrast_loss = self.contrast_loss(Y_2, fc2, contrast_target)
        return out, Y_2, fc2, l,contrast_loss

class PeptideMHCDataset(Dataset):
    """自定义数据集类，实现懒加载和分批处理"""
    def __init__(self, data, global_args, is_train=True):
        """
        data: 原始数据列表
        global_args: 全局参数
        is_train: 是否是训练集
        """
        self.data = data
        self.global_args = global_args
        self.is_train = is_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 预计算长度
        self.length = len(data)

        self.pep_data = [item[0] for item in data]
        self.mhc_data = [item[1] for item in data]
        self.target_data = [item[2] for item in data]
        self.adj_data = [item[3] for item in data]
        self.mhc_seq_data = [item[5] for item in data]
        self.mhc_l_data = [item[6] for item in data]
        self.pep_seq_data = [item[7] for item in data]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """按需加载单个样本，避免全部加载到内存"""
        # item = self.data[idx]
        #
        # # 提取各项数据
        # pep = item[0]
        # mhc = item[1]
        # target = item[2]
        # adj = item[3]
        # mhc_seq = item[5]
        # mhc_l = item[6]
        # pep_seq = item[7]
        return {
            'pep': torch.tensor(self.pep_data[idx], dtype=torch.float32),
            'mhc': torch.tensor(self.mhc_data[idx], dtype=torch.float32),
            'target': torch.tensor(self.target_data[idx], dtype=torch.float32),
            'adj': torch.tensor(self.adj_data[idx], dtype=torch.float32),
            'mhc_seq': torch.tensor(self.mhc_seq_data[idx].squeeze(), dtype=torch.long),
            'mhc_l': torch.tensor(self.mhc_l_data[idx], dtype=torch.long),
            'pep_seq': torch.tensor(self.pep_seq_data[idx].squeeze(), dtype=torch.long),
        }
        # 转换为张量（使用float16节省内存）
        # return {
        #  # 'pep': torch.tensor(pep, dtype=torch.float32),
        #      'pep':torch.tensor(np.array(pep), dtype=torch.float32).clone().detach(),
        #     'mhc': torch.tensor(np.array(mhc), dtype=torch.float32).clone().detach(),
        #     'target': torch.tensor(np.array(target), dtype=torch.float32).clone().detach(),
        #     'adj': torch.tensor(np.array(adj), dtype=torch.float32).clone().detach(),
        #     'mhc_seq': torch.tensor(np.array(mhc_seq).squeeze(), dtype=torch.long).clone().detach(),
        #     # 'mhc_l':mhc_l.clone().detach()
        #     'mhc_l': torch.tensor(np.array(mhc_l), dtype=torch.long).clone().detach(),
        #     'pep_seq': torch.tensor(np.array(pep_seq).squeeze(), dtype=torch.long).clone().detach(),
        # }
    
    # def clear_temp_files(self):
    #     """清理临时文件"""
    #     for f in os.listdir(self.temp_dir):
    #         os.remove(os.path.join(self.temp_dir, f))
    #     os.rmdir(self.temp_dir)

def collate_fn(batch):
    """自定义批次处理函数"""
    return {
        'pep': torch.stack([item['pep'] for item in batch]),
        'mhc': torch.stack([item['mhc'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
        'adj': torch.stack([item['adj'] for item in batch]),
        'mhc_seq': torch.stack([item['mhc_seq'] for item in batch]),
        'mhc_l': torch.stack([item['mhc_l'] for item in batch]),
        'pep_seq': torch.stack([item['pep_seq'] for item in batch]),
    }

def cross_validation_training_transformer_gat(training_data, test_dict, global_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    [blosum_matrix, aa, main_dir, output_path] = global_args
    
    print("Creating datasets...")
    
    # 创建数据集对象（不立即加载数据）
    train_dataset = PeptideMHCDataset(training_data, global_args, is_train=True)
    val_dataset = PeptideMHCDataset(test_dict, global_args, is_train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,  # 使用多进程加载
        pin_memory=True,  # 加速数据传输到GPU
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 立即释放原始数据内存
    # del training_data, test_dict
    # gc.collect()
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 超参数
    fc2_size = 1
    models = []
    num_epochs = 500
    
    # 创建模型
    # 注意：这里需要根据实际输入维度调整
    model = TransformerModel(
        seq1_len=24,  # 假设肽序列长度15，根据实际情况修改
        seq2_len=300,  # 假设MHC序列长度34，根据实际情况修改
        fc2_size=fc2_size
    ).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 0.00005}
    ])
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            # 将批次数据移动到GPU
            pep = batch['pep'].to(device, non_blocking=True)
            mhc = batch['mhc'].to(device, non_blocking=True)
            adj = batch['adj'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            mhc_seq = batch['mhc_seq'].to(device, non_blocking=True)
            pep_seq = batch['pep_seq'].to(device, non_blocking=True)
            mhc_l = batch['mhc_l'].to(device, non_blocking=True)
            # print(mhc_l.size())
            # 前向传播
            optimizer.zero_grad()
            outputs, pep_contrast, mhc_contrast, l,c_l = model(pep, mhc, adj, mhc_seq, pep_seq)
            
            # 计算损失
            # cont_loss = model.contrast_loss(pep_contrast, mhc_contrast)
            loss = criterion(outputs.squeeze(), target) \
                   + model.contrast_weight *model.com_loss(torch.zeros(c_l.size()).view(-1).to(device),c_l.view(-1))\
                   +0.1*compute_loss(l, mhc_l)
            # loss += 0.1 * model.com_loss(l.view(-1, l.size(-1)), mhc_l.view(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        total_val_pcc = 0
        total_val_roc = 0
        total_val_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                # 将批次数据移动到GPU
                pep = batch['pep'].to(device, non_blocking=True)
                mhc = batch['mhc'].to(device, non_blocking=True)
                adj = batch['adj'].to(device, non_blocking=True)
                target = batch['target'].to(device, non_blocking=True)
                mhc_seq = batch['mhc_seq'].to(device, non_blocking=True)
                pep_seq = batch['pep_seq'].to(device, non_blocking=True)
                mhc_l = batch['mhc_l'].to(device, non_blocking=True)
                
                # 前向传播
                outputs, _, _, _,_ = model(pep, mhc, adj, mhc_seq, pep_seq)
                
                # 计算损失
                loss = criterion(outputs.squeeze(), target)
                pcc, roc_auc, max_acc= model_eval(outputs.squeeze(), target)
                
                total_val_loss += loss.item()
                total_val_pcc += pcc
                total_val_roc += roc_auc
                total_val_acc += max_acc
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_pcc = total_val_pcc / len(val_loader)
        avg_val_roc = total_val_roc / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        
        # 打印和记录结果
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val PCC: {avg_val_pcc:.4f}, Val ROC: {avg_val_roc:.4f}, Val Acc: {avg_val_acc:.4f}")
        foutput(str(epoch) + "\t" + str(avg_train_loss) + "\t" + str(avg_val_loss) + "\t" + str(avg_val_pcc) + "\t" + str(
            avg_val_roc) + "\t" + str(avg_val_acc), output_path + 'result.txt')
        # 保存模型检查点（可选）
        if avg_val_pcc > 0.85:  # 根据实际情况调整条件
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            models.append(model)
    
    # 清理临时文件
    train_dataset.clear_temp_files()
    val_dataset.clear_temp_files()
    
    print("Training complete.")
    return models
# 修改训练循环
def cross_validation_training_transformer_gat_0(training_data, test_dict,
                                              global_args):
    # ... 其他部分保持不变 ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    [blosum_matrix, aa, main_dir, output_path] = global_args
    print("Start training data trans")
    training_pepn, training_mhcn, training_targetn, An = [np.array([i[j] for i in training_data]) for j in range(4)]
    training_mhc_seqn, training_mhc_ln, training_pep_seqn =[np.array( [i[j] for i in training_data]) for j in range(5, 8)]
    del training_data
    print("Training data trans over")

    # print(len(training_A),training_A[0])
    # print("数据类型:", type(training_A))
    # print("第一元素类型:", type(training_A[0]))
    # print("第一元素形状:", len(training_A[0]))
    # print("第一行类型:", type(training_A[0][0]))
    # print("第一行长度:", len(training_A[0][0]))
    # validation_pep, validation_mhc = [i[0] for i in validation_data], [i[1] for i in validation_data]

    # 转换为PyTorch Tensor
    training_pep = torch.tensor(training_pepn, dtype=torch.float16, device='cuda')
    del training_pepn

    training_mhc = torch.tensor(training_mhcn, dtype=torch.float16, device='cuda')
    del training_mhcn

    training_target = torch.tensor(training_targetn, dtype=torch.float16, device='cuda')
    del training_targetn

    training_mhc_l = torch.tensor(training_mhc_ln, dtype=torch.float16, device='cuda')
    del training_mhc_ln

    training_pep_seq = torch.tensor(training_pep_seqn.squeeze(), dtype=torch.float16, device='cuda')
    del training_pep_seqn

    training_mhc_seq = torch.tensor(training_mhc_seqn.squeeze(), dtype=torch.float16, device='cuda')
    del training_mhc_seqn
    A =torch.tensor(An, dtype=torch.float16, device='cuda')
    del An
    print("finish training data trans")
    validation_pepn, validation_mhcn, validation_targetn,val_An = [np.array([i[j] for i in test_dict]) for j in range(4)]
    validation_mhc_seqn, validation_mhc_ln, validation_pep_seqn = [np.array([i[j] for i in test_dict]) for j in range(5, 8)]
    del test_dict
    validation_pep = torch.tensor(validation_pepn, dtype=torch.float16, device='cuda')
    del validation_pep
    validation_mhc = torch.tensor(validation_mhcn, dtype=torch.float16, device='cuda')
    del validation_mhcn
    validation_target = torch.tensor(validation_targetn, dtype=torch.float16, device='cuda')
    del validation_targetn
    validation_mhc_l = torch.tensor(validation_mhc_ln, dtype=torch.float16, device='cuda')
    del validation_mhc_ln
    validation_pep_seq = torch.tensor(validation_pep_seqn.squeeze(), dtype=torch.float16, device='cuda')
    del validation_pep_seqn
    validation_mhc_seq = torch.tensor(validation_mhc_seqn.squeeze(), dtype=torch.float16, device='cuda')
    del validation_mhc_seqn
    val_A =torch.tensor(val_An, dtype=torch.float16, device='cuda')
    del val_An
    # validation_pep = torch.cuda.FloatTensor(np.array(validation_pep))
    # validation_mhc = torch.cuda.FloatTensor(np.array(validation_mhc))
    # validation_pep_seq = torch.cuda.LongTensor(np.array(validation_pep_seq).squeeze())
    # validation_mhc_seq = torch.cuda.LongTensor(np.array(validation_mhc_seq).squeeze())
    # validation_target = torch.cuda.FloatTensor(np.array(validation_target))
    # validation_mhc_l = torch.cuda.LongTensor(np.array(validation_mhc_l))
    # val_A = torch.cuda.FloatTensor(np.array(val_A))
    print("finish validation data trans")

    # print(training_mhc_l.size(),validation_mhc_l.size(),validation_target.size())
    # 创建邻接矩阵
    # aa = creat_adj()  # 假设已实现
    # aa = np.expand_dims(aa, axis=0).astype(np.float32)
    # A = np.tile(aa, (len(training_mhc), 1, 1))
    # val_A = np.tile(aa, (len(validation_mhc), 1, 1))
    # A = torch.FloatTensor(A)
    # val_A = torch.FloatTensor(val_A)

    # 超参数
    fc2_size = 1
    models = []
    num_epochs = 150
    # 训练循环

    model = TransformerModel(
        seq1_len=training_pep.shape[1],
        seq2_len=training_mhc.shape[1],
        fc2_size=fc2_size
    ).to(device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.00002}
        # {'params': model.esm_encoder.esm_model.parameters(), 'lr': 1e-5}
    ])
    criterion = nn.MSELoss()
    epoch_plot = []
    train_plot = []
    test_plot = []
    poor_init = False
    val_pcc = 0
    val_roc = 0
    val_acc = 0
    while len(models) == 0:
        for epoch in range(num_epochs):
            print("Begin, epoch:", epoch)
            model.train()
            # 创建DataLoader
            dataset = TensorDataset(training_pep, training_mhc, A, training_target, training_mhc_seq, training_pep_seq,
                                    training_mhc_l)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            total_loss = 0
            t_loss = 0

            # 修改训练步骤
            for batch_pep, batch_mhc, batch_adj, batch_target, batch_mhc_seq, batch_pep_seq, batch_mhc_l in tqdm(loader,
                                                                                                                 desc=f'epoch{epoch + 1}/{num_epochs}'):
                batch_pep = batch_pep.to(device)
                batch_mhc = batch_mhc.to(device)
                batch_adj = batch_adj.to(device)
                batch_target = batch_target.to(device)
                batch_mhc_l = batch_mhc_l.to(device)

                batch_mhc_seq = batch_mhc_seq.to(device)
                batch_pep_seq = batch_pep_seq.to(device)
                # ... 前向传播 ...
                # print(batch_mhc_seq.size(),batch_pep_seq.size())
                optimizer.zero_grad()

                outputs, pep_contrast, mhc_contrast, l = model(batch_pep, batch_mhc, batch_adj, batch_mhc_seq,
                                                               batch_pep_seq)

                # 计算对比损失
                shuffle_idx = torch.randperm(pep_contrast.size(0))
                neg_contrast = pep_contrast[shuffle_idx]
                cont_loss = model.contrast_loss(
                    pep_contrast,
                    mhc_contrast  # 正样本对
                    # neg_contrast    # 负样本
                )
                # cont_loss=simcse_unsup_loss(pep_contrast,mhc_contrast)
                # mlm_loss = compute_loss(mhc_l, batch_mhc_l)
                # 组合损失
                total_loss = criterion(outputs.squeeze(),
                                       batch_target) + model.contrast_weight * cont_loss + 0.1 * model.com_loss(
                    l.view(-1, l.size(-1)), batch_mhc_l.view(-1))
                t_loss += total_loss.item()

                # ... 反向传播 ...
                total_loss.backward()
                optimizer.step()

                # ... 其他计算 ...

            train_loss = t_loss / len(loader)
            # 修改验证步骤
            print(train_loss)

            with torch.no_grad():
                v_loss = 0

                # 验证
                model.eval()
                valdataset = TensorDataset(validation_pep, validation_mhc, val_A, validation_target, validation_mhc_seq,
                                           validation_pep_seq, validation_mhc_l)
                val_loader = DataLoader(valdataset, batch_size=256, shuffle=True)

                for batch_vpep, batch_vmhc, batch_vdj, batch_vtarget, batch_vmhc_seq, batch_vpep_seq, batch_vmhc_l in val_loader:
                    batch_vpep = batch_vpep.to(device)
                    batch_vmhc = batch_vmhc.to(device)
                    batch_vdj = batch_vdj.to(device)
                    batch_vtarget = batch_vtarget.to(device)
                    batch_vmhc_l = batch_vmhc_l.to(device)
                    batch_vmhc_seq = batch_vmhc_seq.to(device)
                    batch_vpep_seq = batch_vpep_seq.to(device)
                    # optimizer.zero_grad()
                    outputs, pep_contrast, mhc_contrast, l = model(batch_vpep, batch_vmhc, batch_vdj, batch_vmhc_seq,
                                                                   batch_vpep_seq)
                    # print('outyput',outputs)

                    # shuffle_idx = torch.randperm(pep_contrast.size(0))
                    # neg_contrast = pep_contrast[shuffle_idx]
                    # cont_loss = model.contrast_loss(
                    #     pep_contrast,
                    #     mhc_contrast,  # 正样本对
                    #     neg_contrast    # 负样本
                    # )
                    # mlm_loss = compute_loss(mhc_l, batch_mhc_l)

                    # 组合损失
                    total_loss = criterion(outputs.squeeze(), batch_vtarget)
                    pcc, roc_auc, max_acc = model_eval(outputs.squeeze(), batch_vtarget)
                    # loss.backward()
                    # optimizer.step()
                    val_pcc += pcc
                    val_roc += roc_auc
                    val_acc += max_acc
                    v_loss += total_loss.item()

                val_loss = v_loss / len(val_loader)
                val_pcc = val_pcc / len(val_loader)
                val_roc = val_roc / len(val_loader)
                val_acc = val_acc / len(val_loader)

                poor_init = True

                # if epoch == 0 and not ((val_pcc< 1) and (val_pcc>0.1)):
                #     poor_init = True
                #     break
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f},Test pcc: {val_pcc:.4f},Test roc: {val_roc:.4f},Test acc: {val_acc:.4f}")
                foutput(str(epoch) + "\t" + str(train_loss) + "\t" + str(val_loss) + "\t" + str(val_pcc) + "\t" + str(
                    val_roc) + "\t" + str(val_acc), output_path + 'result.txt')
                # clear matplotlib plot if already exists
                plt.clf()
                # visualize with matplotlib
                epoch_plot.append(epoch + 1)
                train_plot.append(train_loss)
                test_plot.append(val_loss)
                plt.plot(epoch_plot, train_plot, 'b.-', label='Train')
                plt.plot(epoch_plot, test_plot, 'r.-', label='Test')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss (Lower is Better)')
            # plt.title('Training and Validation Loss')
            # # Set ticks
            # plt.yticks([])
            # plt.xlim(1, num_epochs)
            # plt.xticks(range(1, num_epochs + 1))
            # plt.legend()

            plt.savefig('loss_gsm_esm.png')  # 保存图片 fig.savefig('xx.png') 功能相同
    # if poor_init:
    #               continue
    #           if val_pcc > 0.8:
    #               torch.save(model, 'gsm_esm.pt')
    #               models.append(model)

    print("Training complete.")
    # 模型评估
    # performance_dict = model_performance(models, validation_data, validation_target, aa, global_args)

    return models


# 辅助函数需要重新实现
def model_eval(predictions, targets, threshold=0.5):
    # from torchmetrics import PearsonCorrCoef

    # 确保输入为numpy数组
    # if isinstance(predictions, torch.Tensor):
    pred_np = predictions.cpu().detach().numpy()
    target_np = targets.cpu().detach().numpy()
    print(pred_np)
    # 转换为分类标签（假设任务可以转换为二分类）
    # bin_targets = (targets > threshold).astype(int)
    # bin_preds = (predictions > threshold).astype(int)
    #
    # # 计算皮尔逊相关系数
    # try:
    pcc, _ = pearsonr(pred_np, target_np)
    # except:
    #     pcc = 0.0

    # 计算ROC AUC（需要概率值）
    # try:
    test_label = [1 if aff > 1 - log(500) / log(50000) else 0 for aff in target_np]
    fpr, tpr, thresholds = roc_curve(test_label, pred_np)
    roc_auc = auc(fpr, tpr)
    # except:
    #     roc_auc = 0.5

    # 计算最大准确率（动态阈值）
    threshold = 1 - log(500) / log(50000)
    p = [0 if score < threshold else 1 for score in pred_np]
    accurate = [1 if p[i] == test_label[i] else 0 for i in range(len(p))]
    acc = np.sum(accurate) / float(len(accurate))

    return pcc, roc_auc, acc


def model_performance(models, validation_data, validation_target, aa, global_args):
    [blosum_matrix, _, main_dir, output_path] = global_args
    test_pep, test_mhc = [[i[j] for i in validation_data] for j in range(2)]
    #
    # test_pep = np.array([i[0] for i in test_dict["data"]])
    # test_mhc = np.array([i[1] for i in test_dict["data"]])
    # test_target = np.array(test_dict["target"])
    test_pep = torch.FloatTensor(np.array(test_pep))
    test_mhc = torch.FloatTensor(np.array(test_mhc))
    test_target = torch.FloatTensor(np.array(validation_target)).unsqueeze(1)

    # 创建邻接矩阵
    aa_tensor = torch.FloatTensor(np.tile(aa, (len(test_pep), 1, 1)))

    performance_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            # 预测测试集
            inputs = (test_pep.to(device),
                      test_mhc.to(device),
                      aa_tensor.to(device))
            outputs = model(*inputs)

            # 计算指标
            pcc, roc_auc, max_acc = model_eval(outputs, test_target)

            # 记录结果
            performance_dict[f"model_{i + 1}"] = {
                "PCC": pcc,
                "AUC": roc_auc,
                "Max_Accuracy": max_acc,
                "Predictions": outputs.tolist(),
                "Targets": test_target.tolist()
            }

            # 输出结果
            print(f"Model {i + 1} Performance:")
            print(f"PCC: {pcc:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Max Accuracy: {max_acc:.4f}")
            print("-" * 40)

    return performance_dict  # 辅助函数需要重新实现


def model_eval(predictions, targets, threshold=0.5):
    # from torchmetrics import PearsonCorrCoef

    # 确保输入为numpy数组
    # if isinstance(predictions, torch.Tensor):
    pred_np = predictions.cpu().detach().numpy()
    target_np = targets.cpu().detach().numpy()

    # 转换为分类标签（假设任务可以转换为二分类）
    # bin_targets = (targets > threshold).astype(int)
    # bin_preds = (predictions > threshold).astype(int)
    #
    # # 计算皮尔逊相关系数
    # try:
    pcc, _ = pearsonr(pred_np, target_np)
    # except:
    #     pcc = 0.0

    # 计算ROC AUC（需要概率值）
    # try:
    test_label = [1 if aff > 1 - log(500) / log(50000) else 0 for aff in target_np]
    fpr, tpr, thresholds = roc_curve(test_label, pred_np)
    roc_auc = auc(fpr, tpr)
    # except:
    #     roc_auc = 0.5

    # 计算最大准确率（动态阈值）
    threshold = 1 - log(500) / log(50000)
    p = [0 if score < threshold else 1 for score in pred_np]
    accurate = [1 if p[i] == test_label[i] else 0 for i in range(len(p))]
    acc = np.sum(accurate) / float(len(accurate))

    return pcc, roc_auc, acc


def model_performance(models, validation_data, validation_target, aa, global_args):
    [blosum_matrix, _, main_dir, output_path] = global_args
    test_pep, test_mhc = [[i[j] for i in validation_data] for j in range(2)]
    #
    # test_pep = np.array([i[0] for i in test_dict["data"]])
    # test_mhc = np.array([i[1] for i in test_dict["data"]])
    # test_target = np.array(test_dict["target"])
    test_pep = torch.FloatTensor(np.array(test_pep))
    test_mhc = torch.FloatTensor(np.array(test_mhc))
    test_target = torch.FloatTensor(np.array(validation_target)).unsqueeze(1)

    # 创建邻接矩阵
    aa_tensor = torch.FloatTensor(np.tile(aa, (len(test_pep), 1, 1)))

    performance_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            # 预测测试集
            inputs = (test_pep.to(device),
                      test_mhc.to(device),
                      aa_tensor.to(device))
            outputs = model(*inputs)

            # 计算指标
            pcc, roc_auc, max_acc = model_eval(outputs, test_target)

            # 记录结果
            performance_dict[f"model_{i + 1}"] = {
                "PCC": pcc,
                "AUC": roc_auc,
                "Max_Accuracy": max_acc,
                "Predictions": outputs.tolist(),
                "Targets": test_target.tolist()
            }

            # 输出结果
            print(f"Model {i + 1} Performance:")
            print(f"PCC: {pcc:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Max Accuracy: {max_acc:.4f}")
            print("-" * 40)

    return performance_dict

    # 在模型评估中增加对比学习可视化
def visualize_contrast(contrast_emb, labels, epoch):
        plt.figure(figsize=(10, 8))
        plt.scatter(contrast_emb[:, 0], contrast_emb[:, 1], c=labels)
        plt.colorbar()
        plt.title(f"Contrastive Embeddings (Epoch {epoch})")
        plt.savefig(f"contrast_epoch{epoch}.png")
        plt.close()