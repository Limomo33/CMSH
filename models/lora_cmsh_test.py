# 在模型定义部分新增以下内容
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import log
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
from scipy.stats import spearmanr

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
        for param in self.base_layer.parameters():
            param.requires_grad = False
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        base_output = self.base_layer(x)

        lora_output = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        lora_output = lora_output * self.scaling

        return base_output + lora_output


class ESM2WithLoRA(nn.Module):
    def __init__(self, esm_model, target_modules: Optional[List[str]] = None,
                 r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.model = esm_model

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj"]
        self.modified_layers = []
        self._apply_lora(target_modules, r, lora_alpha, lora_dropout)


    def _apply_lora(self, target_modules: List[str], r: int, lora_alpha: int, lora_dropout: float):
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            is_target = any(target in name for target in target_modules)
            if is_target:
                parent = self._get_parent_module(name)
                child_name = name.split('.')[-1]

                lora_layer = LoRALinear(module, r, lora_alpha, lora_dropout)
                setattr(parent, child_name, lora_layer)
                self.modified_layers.append(name)

    def _get_parent_module(self, module_path: str):
        parts = module_path.split('.')
        current = self.model
        for part in parts[:-1]:  # 
            current = getattr(current, part)
        return current

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"trainable_params: {trainable_params} / total_params: {total_params}")
        print(f"trainable_params: {100 * trainable_params / total_params:.2f}%")
        print(f"modified_layers: {self.modified_layers}")


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

        for param in self.esm_model.parameters():
            param.requires_grad_(False)

        self.repr_layer = self.esm_model.num_layers
        self.embed_dim = self.esm_model.embed_dim

    def forward(self, sequences):

        with torch.no_grad():
            results = self.esm_model(sequences, repr_layers=[self.repr_layer])
        token_embeddings = results["representations"][self.repr_layer]
        contrast_emb = self.contrast_proj(token_embeddings.mean(1))

        # embeddings = self._remove_special_tokens(token_embeddings, sequences)
        return token_embeddings, contrast_emb


class EnhancedESMModelWrapper(nn.Module):
    def __init__(self, esm_model_name: str = "esm2_t6_8M_UR50D",
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(esm_model_name)

        self.esm_model = ESM2WithLoRA(
            esm_model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        self.esm_model.print_trainable_parameters()

        self.contrast_proj = nn.Sequential(
            nn.Linear(esm_model.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, sequences):
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


class AttentionSubgraphExtractor(nn.Module):
    def __init__(self, in_dim, att_dim=64, topk=20):
        super().__init__()
        self.W_q = nn.Linear(in_dim, att_dim)  #
        self.W_k = nn.Linear(in_dim, att_dim)  # 
        self.att_vec = nn.Parameter(torch.randn(2 * att_dim))  # 
        self.topk = topk
        self.leaky_relu = nn.LeakyReLU(0.2)
        adj_template = self.create_adj_template()
        self.register_buffer('A_mp', adj_template)  # 

        # self.A_mp = self.create_adj_template()

    def create_adj_template(self):
        size = 324
        adj = torch.eye(size, dtype=torch.float)  # 

        # 
        indices = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99,
                   114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]

        for row in range(300, 324):
            for col in indices:
                adj[row, col] = 1
                adj[col, row] = 1  # 

        return adj

    def forward(self, H_m, H_p, A_mm):
        """
        in:
            H_m:  [batch, num_mhc, feat_dim]
            H_p:  [batch, num_pep, feat_dim]
            A_mm: MHC-p [batch, num_mhc, num_mhc]
            A_mp: MHC-p [batch, num_mhc, num_pep]

        out:
            A_sub:  [batch, num_mhc, num_mhc + topk]
            H_sub:  [batch, num_mhc, feat_dim]
        """
        batch_size, num_mhc, _ = H_m.size()
        batch_size, num_pep, _ = H_p.size()

        # 1. 
        Q = self.W_q(H_m)  # [b, m, att_dim]
        K = self.W_k(H_p)  # [b, p, att_dim]

        Q_exp = Q.unsqueeze(2)  # [b, m, 1, att_dim]
        K_exp = K.unsqueeze(1)  # [b, 1, p, att_dim]

        att_scores = torch.einsum('bmpd,d->bmp',
                                  torch.cat([Q_exp.expand(-1, -1, num_pep, -1),
                                             K_exp.expand(-1, num_mhc, -1, -1)], dim=-1),
                                  self.att_vec)
        att_scores = F.softmax(att_scores,dim=-1)

        A_mp_sub = self.A_mp[:300, 300:]  # [b, 300, 24]
        fixed_mask = A_mp_sub.unsqueeze(0).expand(batch_size, -1, -1)
        att_mask = (fixed_mask == 0).float() * -1e9
        # print(att_scores.device, att_mask.device)

        masked_att_scores = att_scores + att_mask
        topk_vals, topk_indices = torch.topk(masked_att_scores, k=self.topk, dim=-1)

        A_sub = torch.zeros(batch_size, 324, 324, device=H_m.device)
        A_sub[:, :300, :300] = torch.eye(300)#A_mm[:, :300, :300]
        batch_idx = torch.arange(batch_size).view(-1, 1, 1)  # [b, 1, 1]
        mhc_idx = torch.arange(300).view(1, -1, 1)  # [1, 300, 1]

        # MHC->p
        A_sub[batch_idx, mhc_idx, 300 + topk_indices] = 1.0

        # p->MHC
        A_sub[batch_idx, 300 + topk_indices, mhc_idx] = 1.0
        # print('2222222',A_sub.size())

        # 6.  [b, 324, feat_dim]
        H_sub = torch.cat([H_m, H_p], dim=1)  # 

        # # 
        # for b in range(batch_size):
        #     for m in range(num_mhc):
        #         for idx, p_idx in enumerate(topk_indices[b, m]):
        #             A_sub[b, m, num_mhc + idx] = 1.0
        #
        # # 3. 
        # H_sub = torch.cat([
        #     H_m,
        #     torch.gather(H_p, 1, topk_indices.unsqueeze(-1).expand(-1, -1, -1, H_p.size(-1)))
        # ], dim=1)

        return A_sub, H_sub


class SubgraphGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(GCNConv(in_channels, out_channels))

        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, H, A):
        """
        in:
            H:  [batch_size, num_nodes, input_dim]
            A:  [batch_size, num_nodes, num_nodes]

        out:
             [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, _ = H.size()
        outputs = []

        for i in range(batch_size):
            h = H[i]  # [num_nodes, input_dim]
            adj = A[i]  # [num_nodes, num_nodes]

            edge_index, edge_weight = dense_to_sparse(adj)

            edge_index = edge_index.to(h.device)
            edge_weight = edge_weight.to(h.device)

            for j, layer in enumerate(self.layers):
                h = layer(h, edge_index, edge_weight)

                if j < len(self.layers) - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)

                    h = self.batch_norm(h)

            outputs.append(h)

        return torch.stack(outputs, dim=0)


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
        attention = attention * adj  # [:attention.size()[0], :, :]
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
        
        super(BatchGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        adj_template = self.create_adj_template()
        edge_index, edge_weight = dense_to_sparse(adj_template)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(GCNConv(in_channels, out_channels, add_self_loops=False))

        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.init_weights()

    def create_adj_template(self):
        size = 324
        adj = torch.eye(size, dtype=torch.float)  # 

        indices = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99,
                   114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]

        for row in range(300, 324):
            for col in indices:
                adj[row, col] = 1
                adj[col, row] = 1  # 

        return adj

    def init_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'lin'):  # 
                nn.init.xavier_uniform_(layer.lin.weight)
                if layer.lin.bias is not None:
                    nn.init.zeros_(layer.lin.bias)

    def forward(self, x):
       
        batch_size, num_nodes, _ = x.size()

        batch_outputs = []

        for i in range(batch_size):
            sample_x = x[i]  # (num_nodes, input_dim)


            for j, layer in enumerate(self.layers):
                sample_x = layer(sample_x, self.edge_index, self.edge_weight)

                if j < len(self.layers) - 1:
                    sample_x = F.leaky_relu(sample_x, negative_slope=0.01)
                    sample_x = F.dropout(sample_x, p=self.dropout, training=self.training)

                    sample_x = self.batch_norm(sample_x)

            batch_outputs.append(sample_x)

        return torch.stack(batch_outputs, dim=0)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, n_final_out, dropout=0.5, alpha=0.2, nheads=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([

            GraphAttentionLayer(

                in_features=nfeat,  # 

                out_features=nhid,  # 

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
    mhc_con = F.normalize(mhc_con, dim=1)
    pep_con = F.normalize(pep_con, dim=1)
    batch_size = mhc_con.size(0)
    labels = torch.arange(batch_size, device=mhc_con.device)
    sim_matrix = mhc_con @ pep_con.T  # [batch_size, batch_size]
    sim_matrix /= temperature

    loss_mhc_to_peptide = F.cross_entropy(sim_matrix, labels)
    loss_peptide_to_mhc = F.cross_entropy(sim_matrix.T, labels)

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



class TrainableGaussianNoise(nn.Module):


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
        self.register_buffer('B', torch.zeros(324))

        if trainable:
            if per_channel:
                self.log_std = nn.Parameter(torch.zeros(1) + torch.log(torch.tensor(initial_std)))
            else:
                self.log_std = nn.Parameter(torch.zeros(1) + torch.log(torch.tensor(initial_std)))
        else:
            self.register_buffer('log_std', torch.tensor(initial_std).log())

    def gnm_b_factor(self, A):
        connectivity=A[0,:,:]
        # print(connectivity.size())
        N = connectivity.shape[0]

        T = -connectivity
        diag_sum = -torch.sum(T, dim=1)  # 
        T = T + torch.diag(diag_sum)

        try:
            T_inv = torch.linalg.pinv(T)
        except torch.linalg.LinAlgError:
            T_inv = torch.linalg.pinv(T + 1e-6 * torch.eye(N))

        diag_T_inv = torch.diag(T_inv)

        b_factors = diag_T_inv / torch.max(diag_T_inv) * 80.0  # 

        return b_factors

    def bfactor_to_noise(self,A, sigma_min=0.05, sigma_max=0.30, B_m=50.0, k=0.1):
      
        b_factor = self.gnm_b_factor(A)
        sigmoid = 1 / (1 + torch.exp(-k * (b_factor - B_m)))
        sigma = sigma_min + (sigma_max - sigma_min) * sigmoid

        return torch.clamp(sigma, min=sigma_min, max=sigma_max)
    def forward(self, x):
        # print('111111111111',x.size())
        self.B = self.bfactor_to_noise(x)
        # self.register_buffer('B', B)  # 
        if self.training or self.always_active:
            # std1 = torch.exp(self.log_std)
            # 
            if self.per_channel and x.dim() > 2:
                noise_shape = [1] * x.dim()
                noise_shape[1] = x.size(1)  # 
                noise = torch.randn(*noise_shape, device=x.device) * std
            else:
                noise = torch.randn_like(x) * std

            # 
            x = x + noise

            if self.clamp_output is not None:
                x = torch.clamp(x, *self.clamp_output)

        return x

    def current_std(self):
        """"""
        return torch.exp(self.log_std).item()

    def extra_repr(self):
        """"""
        return (f"std={self.current_std():.4f}, per_channel={self.per_channel}, "
                f"trainable={self.trainable}, always_active={self.always_active}")


class TrainableGaussianNoiseseq(nn.Module):
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

        # 
        if trainable:
            if per_channel:
                self.log_std = nn.Parameter(torch.zeros(1) + torch.log(torch.tensor(initial_std)))
            else:
                self.log_std = nn.Parameter(torch.zeros(1) + torch.log(torch.tensor(initial_std)))
        else:
            self.register_buffer('log_std', torch.tensor(initial_std).log())

    def forward(self, x):
        if self.training or self.always_active:
            std = torch.exp(self.log_std)

            if self.per_channel and x.dim() > 2:
                noise_shape = [1] * x.dim()
                noise_shape[1] = x.size(1)  # 
                noise = torch.randn(*noise_shape, device=x.device) * std
            else:
                noise = torch.randn_like(x) * std

            x = x + noise

            if self.clamp_output is not None:
                x = torch.clamp(x, *self.clamp_output)

        return x

    def current_std(self):
        return torch.exp(self.log_std).item()

    def extra_repr(self):
        return (f"std={self.current_std():.4f}, per_channel={self.per_channel}, "
                f"trainable={self.trainable}, always_active={self.always_active}")


class TransformerModel(nn.Module):
    def __init__(self, seq1_len, seq2_len, dim=340, kernel_size=32, num_heads=2, gat_heads=8, fc2_size=1):
        super().__init__()
        self.pos_embed1 = PositionEmbeddingTrain(seq1_len, 20)
        self.pos_embed2 = PositionEmbeddingTrain(seq2_len, 20)

        self.attention1 = MultiHeadAttention(340)
        self.attention2 = MultiHeadAttention(340)

        self.gat = GAT(nfeat=20, nhid=128, n_final_out=20)
        self.gconv1 = BatchGCN(20, 128, 32)
        self.linearmhc = nn.Linear(117760, 5120)
        self.linearmhc2 = nn.Linear(5120, 1280)

        self.linear0 = nn.Linear(20736, 2560)  # 

        self.linear1 = nn.Linear(8960, 1280)  # 
        self.linear2 = nn.Linear(3200, 256)  # 
        self.linear3 = nn.Linear(27200, 2560)  # 
        self.linear4 = nn.Linear(3840, 2560)  # 
        self.linear5 = nn.Linear(5120, 2560)
        self.linear6 = nn.Linear(2560, 256)
        self.linear7 = nn.Linear(256, 1)

        # self.linearc = nn.Linear(2560, 300)
        self.linea1 = nn.Linear(7280, 128)  # 
        self.linea2 = nn.Linear(2560, 128)  # 
        self.linea3 = nn.Linear(1920, 128)  # 
        self.linea4 = nn.Linear(384, 2560)  # 
        # self.linea5 = nn.Linear(5120, 2560)
        # self.linea6 = nn.Linear(2560, 256)
        # self.linea7 = nn.Linear(256, 1)

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
            out_channels=64,
            kernel_size=6,
            stride=1,
            padding='same'
        )
        self.conv3 = nn.Conv1d(
            in_channels=300,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.convp2 = nn.Conv1d(
            in_channels=24,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.conv4 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.flatten = nn.Flatten()  # 

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AvgPool1d(kernel_size=1)
        self.pool3 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(0.5)
        # self.linear=nn.Linear(out_features=2)
        self.norm = nn.LayerNorm(320, eps=1e-5, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(2560, eps=1e-5, elementwise_affine=True)

        self.esm_encoder = EnhancedESMModelWrapper()
        self.esm_encoder2 = ESMModelWrapper()
        self.fixed_noise = TrainableGaussianNoise(
            initial_std=0.1,
            trainable=True,
            always_active=True,
            clamp_output=(0, 1))
        self.fixed_noise2 = TrainableGaussianNoiseseq(
            initial_std=0.2,
            trainable=True,
            always_active=True,
            clamp_output=(0, 1))
        # self.contrast_loss = nn.L1Loss(reduction='mean')
        self.contrast_loss = nn.CosineEmbeddingLoss()
        self.contrast_weight = 0.3  # 

        self.com_loss = nn.CrossEntropyLoss(ignore_index=-100)

        self.subgraph_extractor = AttentionSubgraphExtractor(
            in_dim=20,
            att_dim=32,
            topk=15
        )

        self.subgraph_gcn = SubgraphGCN(
            input_dim=20,
            hidden_dim=64,
            output_dim=32
        )
        self.bnn = nn.BatchNorm1d(num_features=324, eps=1e-5, momentum=0.1, affine=True)


        self.bn = nn.BatchNorm1d(num_features=2560, eps=1e-5, momentum=0.1, affine=True)
        self.bn2 = nn.BatchNorm1d(num_features=117760, eps=1e-5, momentum=0.1, affine=True)

        self.bn3 = nn.BatchNorm1d(num_features=8960, eps=1e-5, momentum=0.1, affine=True)
        self.bn4 = nn.BatchNorm1d(num_features=27200, eps=1e-5, momentum=0.1, affine=True)
        self.bn5 = nn.BatchNorm1d(num_features=3840, eps=1e-5, momentum=0.1, affine=True)
    def forward(self, pep, mhc, adj, mhc_seq, pep_seq):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pepesm, pep_contrast = self.esm_encoder2(pep_seq)
        mhcesm, mhc_contrast, l = self.esm_encoder(mhc_seq)

        pepesm = torch.cat([pepesm, pepesm], dim=1)  # 
        # mhc_embeddings = self._pad_sequences(mhc_embeddings, target_len=300)  # 
        # pepesm = self.pos_embed1(pep_embeddings)
        # mhcesm = self.pos_embed2(mhc_embeddings)
        pepesm = self.fixed_noise2(pepesm)

        # mhcesm = self.dropout(mhcesm)

        # print(pep.size(),mhc.size())

        # pep_attn = self.attention1(pepesm)
        # mhc_attn = self.attention2(mhcesm)  # (batch,seq,dim)
        #
        pep_pool = self.pool2(pepesm)
        mhc_pool = self.pool2(mhcesm)
        # # print('11111',mhc_pool.size())
        # mhc_pool = self.dropout(mhc_pool)
        mhc_pool = self.norm(mhc_pool)
        pep_pool = self.norm(pep_pool)

        # adj = self.fixed_noise(adj)
        Y = self.gat(mhc, adj)
        # Y=self.dropout(Y)
        A_sub, H_sub = self.subgraph_extractor(Y, pep, adj)
        # print(A_sub.size())
        H_sub=self.bnn(H_sub)
        A_sub = self.fixed_noise(A_sub)
        # A_sub=self.bnn(A_sub)

        A1 = torch.triu(A_sub) + torch.triu(A_sub, 1).transpose(-1, -2)
        A2 = torch.tril(A_sub) + torch.tril(A_sub, 1).transpose(-1, -2)
        # print(A_sub,H_sub)
        subgraph_features1 = self.subgraph_gcn(H_sub, A1)
        subgraph_features2 = self.subgraph_gcn(H_sub, A2)
        combined = torch.cat([subgraph_features1, subgraph_features2], dim=2)

        # print(subgraph_features)
        # combined = torch.cat([A_sub, pep], dim=1)
        #
        # # y2=torch.cat([mhc,flat_mhc_0],dim=1)
        # Y2=self.gconv1(combined)

        # Y=torch.mul(gat_out,gcn_out)
        # Y = self.pool(gat_out).squeeze(-1)
        Y3 = self.flatten(combined)
        Y3 = self.linear0(Y3)
        Y3 = self.dropout(Y3)
        Y3=self.bn(Y3)
        Y_2 = F.tanh(Y3)
        # print(Y_2)
        # Y_cont = self.linearc(Y_2)
        mhcc = torch.cat([mhc_pool, mhc], dim=2)
        pepp = torch.cat([pep_pool, pep], dim=2)
        mhc_conv1 = self.conv1(mhcc)
        mhc_conv1 = self.pool3(mhc_conv1)
        mhc_conv = self.conv2(mhc_conv1)
        mhc_conv = self.pool3(mhc_conv)
        mhc_conv = self.dropout(mhc_conv)

        pep_conv = self.convp(pepp)
        pep_conv = self.pool3(pep_conv)
#
#         mhc_conv3 = self.conv3(mhc)
#         mhc_conv3=self.pool3(mhc_conv3)
#         mhc_conv4 = self.conv2(mhc_conv3)
#         mhc_conv4=self.pool3(mhc_conv4)
#
#         pep_conv2 = self.convp2(pep)
#         pep_conv2=self.pool3(pep_conv2)
#
#         flat_pep = self.flatten(pep_conv2)
#         flat_mhc_0 = self.flatten(mhc)
#         flat_mhc_1 = self.flatten(mhc_conv3)
#         flat_mhc_2 = self.flatten(mhc_conv4)
#         # print(flat_pep.size(),flat_mhc_1.size(),flat_mhc_2.size())
# #torch.Size([256, 1280]) torch.Size([256, 1280]) torch.Size([256, 640])
#         cat_0 = torch.cat([flat_pep, flat_mhc_0],dim=1)
#         cat_1 = torch.cat([flat_pep, flat_mhc_1],dim=1)
#         cat_2 = torch.cat([flat_pep, flat_mhc_2],dim=1)
#         # print(cat_0.size(),cat_1.size(),cat_2.size())
#         #torch.Size([256, 7280]) torch.Size([256, 2560]) torch.Size([256, 1920])
#         fc1_0 = self.linea1(cat_0)
#         fc1_0 = F.relu(fc1_0)
#         fc1_1 = self.linea2(cat_1)
#         fc1_1 = F.relu(fc1_1)
#         fc1_2 = self.linea3(cat_2)
#         fc1_2 = F.relu(fc1_2)
#         merge_0 = torch.cat([fc1_0, fc1_1, fc1_2],dim=1)
#         fc0_2 = self.linea4(merge_0)
#         fc0_2 = F.relu(fc0_2)

        flat_pep_0 = self.flatten(pep_pool)
        flat_pep_1 = self.flatten(pep_conv)
        flat_mhc_0 = self.flatten(mhc_pool)  #
        # print('11111',mhc_conv1.size(), mhc_conv.size(),pep_conv.size())  # (batch,seq,dim)
        # 11111 torch.Size([128, 128, 180]) torch.Size([128, 64, 90]) torch.Size([128, 128, 180])
        flat_mhc_1 = self.flatten(mhc_conv1)  # .squeeze(-1)
        # print('22222222', flat_mhc_1.size())  # (batch,seq,dim)
        flat_mhc_2 = self.flatten(mhc_conv)  # .squeeze(-1)

        cat_0 = torch.cat([flat_mhc_1, flat_mhc_0], dim=1)
        # print('11111', cat_0.size())
        cat_0 = self.bn2(cat_0)

        cat_0 = self.linearmhc(cat_0)
        cat_0 = F.relu(cat_0)
        cat_0 = self.dropout(cat_0)
        cat_0 = self.linearmhc2(cat_0)
        cat_0 = F.relu(cat_0)
        cat_0 = self.dropout(cat_0)
        cat_1 = torch.cat([flat_pep_0, cat_0], dim=1)
        # # print('222222', cat_1.size())

        cat_2 = torch.cat([flat_pep_1, flat_mhc_2], dim=1)
        # print('111333331', cat_2.size())
        cat_1 = self.bn3(cat_1)

        fc1_0 = self.linear1(cat_1)
        fc1_0 = F.relu(fc1_0)
        fc1_0 = self.dropout(fc1_0)

        # # fc1_1 = self.linear2(cat_1)
        # # fc1_1=F.relu(fc1_1)
        cat_2 = self.bn4(cat_2)

        fc1_2 = self.linear3(cat_2)
        fc1_2 = F.relu(fc1_2)
        fc1_2 = self.dropout(fc1_2)

        merge_1 = torch.cat([fc1_0, fc1_2], dim=1)
        # # print('4444444', merge_1.size())
        merge_1 = self.bn5(merge_1)

        fc2 = self.linear4(merge_1)  # merge_1
        fc2 = F.relu(fc2)
        fc2 = self.dropout(fc2)

        # merge_1_1 = torch.cat([fc2, fc0_2], dim=1)
        # fc2 = self.linear5(merge_1_1)  # merge_1
        # fc2 = F.relu(fc2)
        # Y_2 = self.norm2(Y_2)

        merge_2 = torch.cat([fc2, Y_2], dim=1)
        # fc2=self.norm1(fc2)
        # print('5555555', merge_2.size())

        fc3 = self.linear5(merge_2)  # merge_1
        fc3 = F.tanh(fc3)
        fc3 = self.dropout(fc3)

        fc3 = self.linear6(fc3)  # merge_1
        fc3 = F.tanh(fc3)
        fc3 = self.dropout(fc3)

        out = self.linear7(fc3)
        out = F.leaky_relu(out, negative_slope=0.01)

        contrast_target = torch.ones(Y_2.size(0), device=Y_2.device)
        contrast_loss = self.contrast_loss(Y_2, fc2, contrast_target)
        return out, Y_2, fc2, l, contrast_loss


class PeptideMHCDataset_select(Dataset):

    def __init__(self, data, global_args, is_train=True):
        """
        data: 
        global_args: 
        is_train: 
        """
        self.data = data
        self.global_args = global_args
        self.is_train = is_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pep_len_list = [item[4] for item in data]
        #
        valid_indices = [idx for idx, pep_len in enumerate(pep_len_list) if pep_len == 10]
        # 
        self.pep_data = [data[idx][0] for idx in valid_indices]
        self.mhc_data = [data[idx][1] for idx in valid_indices]
        self.target_data = [data[idx][2] for idx in valid_indices]
        self.adj_data = [data[idx][3] for idx in valid_indices]
        self.mhc_seq_data = [data[idx][5] for idx in valid_indices]
        self.mhc_l_data = [data[idx][6] for idx in valid_indices]
        self.pep_seq_data = [data[idx][7] for idx in valid_indices]
        # self.pep_len = [9] * len(valid_indices)  # 

        # 
        self.length = len(valid_indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'pep': torch.tensor(self.pep_data[idx], dtype=torch.float32),
            'mhc': torch.tensor(self.mhc_data[idx], dtype=torch.float32),
            'target': torch.tensor(self.target_data[idx], dtype=torch.float32),
            'adj': torch.tensor(self.adj_data[idx], dtype=torch.float32),
            'mhc_seq': torch.tensor(self.mhc_seq_data[idx].squeeze(), dtype=torch.long),
            'mhc_l': torch.tensor(self.mhc_l_data[idx], dtype=torch.long),
            'pep_seq': torch.tensor(self.pep_seq_data[idx].squeeze(), dtype=torch.long),
            # 'pep_len': torch.tensor(9, dtype=torch.long),  # 
        }


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
        # 'pep_len': torch.stack([item['pep_len'] for item in batch]),
    }
class PeptideMHCDataset(Dataset):

    def __init__(self, data, global_args, is_train=True):
  
        self.data = data
        self.global_args = global_args
        self.is_train = is_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 
        self.length = len(data)

        self.pep_data = [item[0] for item in data]
        self.mhc_data = [item[1] for item in data]
        self.target_data = [item[2] for item in data]
        self.adj_data = [item[3] for item in data]
        self.mhc_seq_data = [item[5] for item in data]
        self.mhc_l_data = [item[6] for item in data]
        self.pep_seq_data = [item[7] for item in data]
        self.pep_len=[item[4] for item in data]
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # item = self.data[idx]
        #
        # # 
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
            'pep_len': torch.tensor(self.pep_seq_data[idx].squeeze(), dtype=torch.long),
        }

def collate_fn(batch):
    """"""
    return {
        'pep': torch.stack([item['pep'] for item in batch]),
        'mhc': torch.stack([item['mhc'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
        'adj': torch.stack([item['adj'] for item in batch]),
        'mhc_seq': torch.stack([item['mhc_seq'] for item in batch]),
        'mhc_l': torch.stack([item['mhc_l'] for item in batch]),
        'pep_seq': torch.stack([item['pep_seq'] for item in batch]),
        # 'pep_len': torch.stack([item['pep_len'] for item in batch]),
    }


def cross_validation_test(training_data, test_dict, global_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    [blosum_matrix, aa, main_dir, output_path] = global_args

    print("Creating datasets...")

    # train_dataset = PeptideMHCDataset(training_data, global_args, is_train=True)
    val_dataset = PeptideMHCDataset_select(test_dict, global_args, is_train=False)


    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    fc2_size = 1
    models = []
    epoch_plot = []
    train_plot = []
    test_plot = []
    num_epochs = 500
    MAX=0
 
    model = TransformerModel(
        seq1_len=24, 
        seq2_len=300,  
    ).to(device)

        # 验证阶段
    model.load_state_dict(torch.load(output_path+f'model_epoch_120.pth'), strict=False)

    model.eval()
    total_val_loss = 0
    total_val_pcc = 0
    total_val_roc = 0
    total_val_acc = 0
    total_val_srcc = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            # 
            pep = batch['pep'].to(device, non_blocking=True)
            mhc = batch['mhc'].to(device, non_blocking=True)
            adj = batch['adj'].to(device, non_blocking=True)
            # pep_len=batch['pep_len'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            mhc_seq = batch['mhc_seq'].to(device, non_blocking=True)
            pep_seq = batch['pep_seq'].to(device, non_blocking=True)
            mhc_l = batch['mhc_l'].to(device, non_blocking=True)

            # 
            outputs, _, _, _, _ = model(pep, mhc, adj, mhc_seq, pep_seq)

            # 
            # loss = criterion(outputs.squeeze(), target)
            pcc, roc_auc, max_acc, SRCC = model_eval(outputs.squeeze(), target)
            # SRCC, _ = spearmanr(outputs.squeeze(), target)
            # total_val_loss += loss.item()
            total_val_pcc += pcc
            total_val_roc += roc_auc
            total_val_acc += max_acc
            total_val_srcc += SRCC
    if len(val_loader)!=0:
        avg_val_srcc = total_val_srcc / len(val_loader)
        # avg_val_loss = total_val_loss / len(val_loader)
        avg_val_pcc = total_val_pcc / len(val_loader)
        avg_val_roc = total_val_roc / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        # 
    else:
        avg_val_srcc = total_val_srcc
        avg_val_pcc = total_val_pcc
        avg_val_roc = total_val_roc
        avg_val_acc = total_val_acc 
    # 
    print(
          f"Val PCC: {avg_val_pcc:.4f}, Val ROC: {avg_val_roc:.4f}, Val Acc: {avg_val_acc:.4f}, Val SRCC: {avg_val_srcc:.4f}")
    foutput(str(avg_val_pcc) + "\t" + str(
            avg_val_roc) + "\t" + str(avg_val_acc)+ "\t" + str(avg_val_srcc), output_path + 'result_val.txt')
    # plt.clf()
            # visualize with matplotlib
    print("Training complete.")



# 
def model_eval(predictions, targets, threshold=0.5):

    pred_np = predictions.cpu().detach().numpy()
    target_np = targets.cpu().detach().numpy()
  
    n_samples = pred_np.size

    if pred_np.ndim == 0 or target_np.ndim == 0:
        print("Warning: Empty predictions or targets!")
        return 0.0, 0.5, 0.5, 0.0  # 
    if n_samples >= 2:
        try:
            pcc, _ = pearsonr(pred_np, target_np)
        except Exception as e:
            print(f"Error calculating Pearson: {e}")
            pcc = 0.0
    else:
        pcc = 0.0  # 
    
    if n_samples >= 2:
        try:
            SRCC, _ = spearmanr(pred_np, target_np)
        except Exception as e:
            print(f"Error calculating Spearman: {e}")
            SRCC = 0.0
    else:
        SRCC = 0.0
    
    try:
        threshold_val = 1 - log(500) / log(50000)
        test_label = np.array([1 if aff > threshold_val else 0 for aff in target_np])
        
        if n_samples >= 2 and len(np.unique(test_label)) >= 2:
            fpr, tpr, thresholds = roc_curve(test_label, pred_np)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0.5
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        roc_auc = 0.5
    
    try:
        p = np.array([0 if score < threshold_val else 1 for score in pred_np])
        accurate = np.array([1 if p[i] == test_label[i] else 0 for i in range(n_samples)])
        acc = np.sum(accurate) / float(n_samples)
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        acc = 0.5

    test_label = [1 if aff > 1 - log(500) / log(50000) else 0 for aff in target_np]
    fpr, tpr, thresholds = roc_curve(test_label, pred_np)
    roc_auc = auc(fpr, tpr)
    # except:
    #     roc_auc = 0.5

    threshold = 1 - log(500) / log(50000)
    p = [0 if score < threshold else 1 for score in pred_np]
    accurate = [1 if p[i] == test_label[i] else 0 for i in range(len(p))]
    acc = np.sum(accurate) / float(len(accurate))
    # else:
    #     pcc=0
    #     return pcc,pcc,pcc,pcc

    return pcc, roc_auc, acc,SRCC


def model_performance(models, validation_data, validation_target, aa, global_args):
    [blosum_matrix, _, main_dir, output_path] = global_args
    test_pep, test_mhc = [[i[j] for i in validation_data] for j in range(2)]
 
    test_pep = torch.FloatTensor(np.array(test_pep))
    test_mhc = torch.FloatTensor(np.array(test_mhc))
    test_target = torch.FloatTensor(np.array(validation_target)).unsqueeze(1)

    aa_tensor = torch.FloatTensor(np.tile(aa, (len(test_pep), 1, 1)))

    performance_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            # 
            inputs = (test_pep.to(device),
                      test_mhc.to(device),
                      aa_tensor.to(device))
            outputs = model(*inputs)

            # 
            pcc, roc_auc, max_acc = model_eval(outputs, test_target)

            # 
            performance_dict[f"model_{i + 1}"] = {
                "PCC": pcc,
                "AUC": roc_auc,
                "Max_Accuracy": max_acc,
                "Predictions": outputs.tolist(),
                "Targets": test_target.tolist()
            }

            # 
            print(f"Model {i + 1} Performance:")
            print(f"PCC: {pcc:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Max Accuracy: {max_acc:.4f}")
            print("-" * 40)

    return performance_dict  

def visualize_contrast(contrast_emb, labels, epoch):
    plt.figure(figsize=(10, 8))
    plt.scatter(contrast_emb[:, 0], contrast_emb[:, 1], c=labels)
    plt.colorbar()
    plt.title(f"Contrastive Embeddings (Epoch {epoch})")
    plt.savefig(f"contrast_epoch{epoch}.png")

    plt.close()
