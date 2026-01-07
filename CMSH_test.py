
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
from lora_gsm_test import *

def cross_validation_test(training_data, test_dict, global_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    [blosum_matrix, aa, main_dir, output_path] = global_args

    print("Creating datasets...")

    # 创建数据集对象（不立即加载数据）
    train_dataset = PeptideMHCDataset(training_data, global_args, is_train=True)
    val_dataset = PeptideMHCDataset_select(test_dict, global_args, is_train=False)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
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
    epoch_plot = []
    train_plot = []
    test_plot = []
    num_epochs = 500
    MAX = 0
    # 创建模型
    # 注意：这里需要根据实际输入维度调整
    model = TransformerModel(
        seq1_len=24,  # 假设肽序列长度15，根据实际情况修改
        seq2_len=300,  # 假设MHC序列长度34，根据实际情况修改
        fc2_size=fc2_size
    ).to(device)

    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 0.00001, 'weight_decay': 0.001}
    ])
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # 训练阶段
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
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
            outputs, pep_contrast, mhc_contrast, l, c_l = model(pep, mhc, adj, mhc_seq, pep_seq)

            # 计算损失
            # cont_loss = model.contrast_loss(pep_contrast, mhc_contrast)
            loss = criterion(outputs.squeeze(), target) \
                   + 0.1 * c_l \
                   + 0.1 * compute_loss(l, mhc_l)
            # loss += 0.1 * model.com_loss(l.view(-1, l.size(-1)), mhc_l.view(-1))

            loss.backward()
            optimizer.step()
            foutput(str(epoch) + "\t" + str(loss), output_path + 'train_loss.txt')
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段

        model.eval()
        total_val_loss = 0
        total_val_pcc = 0
        total_val_roc = 0
        total_val_acc = 0
        total_val_srcc = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                # 将批次数据移动到GPU
                pep = batch['pep'].to(device, non_blocking=True)
                mhc = batch['mhc'].to(device, non_blocking=True)
                adj = batch['adj'].to(device, non_blocking=True)
                # pep_len=batch['pep_len'].to(device, non_blocking=True)
                target = batch['target'].to(device, non_blocking=True)
                mhc_seq = batch['mhc_seq'].to(device, non_blocking=True)
                pep_seq = batch['pep_seq'].to(device, non_blocking=True)
                mhc_l = batch['mhc_l'].to(device, non_blocking=True)

                # 前向传播
                outputs, _, _, _, _ = model(pep, mhc, adj, mhc_seq, pep_seq)

                # 计算损失
                loss = criterion(outputs.squeeze(), target)
                pcc, roc_auc, max_acc, SRCC = model_eval(outputs.squeeze(), target)
                # SRCC, _ = spearmanr(outputs.squeeze(), target)
                total_val_loss += loss.item()
                total_val_pcc += pcc
                total_val_roc += roc_auc
                total_val_acc += max_acc
                total_val_srcc += SRCC
        avg_val_srcc = total_val_srcc / len(val_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_pcc = total_val_pcc / len(val_loader)
        avg_val_roc = total_val_roc / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        # 反向传播

        # 打印和记录结果
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val PCC: {avg_val_pcc:.4f}, Val ROC: {avg_val_roc:.4f}, Val Acc: {avg_val_acc:.4f}, Val SRCC: {avg_val_srcc:.4f}")
        foutput(
            str(epoch) + "\t" + str(avg_train_loss) + "\t" + str(avg_val_loss) + "\t" + str(avg_val_pcc) + "\t" + str(
                avg_val_roc) + "\t" + str(avg_val_acc) + "\t" + str(avg_val_srcc), output_path + 'result_9mer.txt')
        plt.clf()
        # visualize with matplotlib
        epoch_plot.append(epoch + 1)
        train_plot.append(avg_train_loss)
        test_plot.append(avg_val_loss)
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

        plt.savefig('loss_gsm_9.png')  # 保存图片 fig.savefig('xx.png') 功能相同
        # 保存模型检查点（可选）
        if epoch == 120:  # 根据实际情况调整条件
            torch.save(model.state_dict(), output_path + f"model_epoch_120.pth")
        if avg_val_pcc > MAX and avg_val_pcc > 0.87:  # 根据实际情况调整条件
            MAX = avg_val_pcc
            torch.save(model.state_dict(), output_path + f"model_epoch_{epoch + 1}.pth")
            models.append(model)

    # 清理临时文件
    train_dataset.clear_temp_files()
    val_dataset.clear_temp_files()

    print("Training complete.")
    return models