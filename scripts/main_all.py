import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import re
from math import log
import numpy as np
import random
from sklearn.model_selection import KFold
from lora_cmsh_test import *
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PretrainedConfig, PreTrainedModel
import esm
import os
from adj import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def read_blosum(path):
    '''
    Read the blosum matrix from the file blosum50.txt
    Args:
        1. path: path to the file blosum50.txt
    Return values:
        1. The blosum50 matrix
    '''
    f = open(path, "r")
    blosum = []
    for line in f:
        blosum.append([(float(i)) / 10 for i in re.split("\t", line)])
        # The values are rescaled by a factor of 1/10 to facilitate training
    f.close()
    # print(blosum)
    return blosum


def foutput(content, output_path):
    content = str(content)
    fout = open(output_path, "a+")
    fout.write(content + "\n")
    fout.close()


def allele_seq(path):
    seq_dict = {}
    f = open(path, "r")
    l = ''
    allele = None
    for line in f:
        if line[0] == ">":  # A new allele
            match = re.search("(\S+\*\d+:\d+)", line)  # The standard allele id are like "A*01:01:..."
            allele = None  # While reading the sequence of the same alleles from different
            if match != None:  # If the current allele has a name with the correct format
                if match.groups()[0] not in seq_dict.keys():
                    allele = match.groups()[0]  # A new allele
                    seq_dict[allele] = ""  # And its sequence
        elif allele != None:
            seq_dict[allele] = seq_dict[allele] + line[:-1]
            # Each line contains only 60 redidues, so add the sequence of the current line
            # to the end of the corresponding sequence
    for allele in list(seq_dict.keys()):
        if len(seq_dict[allele]) < 362:
            seq_dict.pop(allele)
    return seq_dict


def pseudo_seq_transformer(seq_dict, global_args):
    [blosum_matrix, aa, main_dir, output_path] = global_args

    pseq_dict = {}  # pseudo sequence dictionary
    residue_indices = [i for i in range(300)]

    for allele in seq_dict.keys():
        new_pseq = []
        pseq = ""
        for index in residue_indices:
            pseq += seq_dict[allele][index]
            new_pseq.append(
                blosum_matrix[aa[seq_dict[allele][index]]])  # +[i for i in pp_matrix[aa[seq_dict[allele][index]]]])
        pseq_dict[allele] = new_pseq
    return pseq_dict


def pseudo_seq_esm(seq_dict, global_args):
    [blosum_matrix, aa, main_dir, output_path] = global_args
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pseq_dict = {}  # pseudo sequence dictionary
    residue_indices = [i for i in range(300)]
    new_dir = '/root/ckpt'
    tokenizer = AutoTokenizer.from_pretrained(new_dir)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15

    )
    for allele in seq_dict.keys():
        new_pseq = []
        pseq = ""
        for index in residue_indices:
            pseq += seq_dict[allele][index]
            new_pseq.append(
                blosum_matrix[aa[seq_dict[allele][index]]])  # +[i for i in pp_matrix[aa[seq_dict[allele][index]]]])
        #     print(collator(tokenizer(
        #     pseq,
        #     max_length=300,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="np"  # 确保返回numpy数组

        # )['input_ids']))

        if allele not in pseq_dict.keys():
            pseq_dict[allele] = [new_pseq, collator(tokenizer(
                pseq,
                max_length=300,
                padding="max_length",
                truncation=True,
                return_tensors="np"  # 确保返回numpy数组

            )['input_ids'])['input_ids'], collator(tokenizer(
                pseq,
                max_length=300,
                padding="max_length",
                truncation=True,
                return_tensors="np"  # 确保返回numpy数组

            )['input_ids'])['labels']]
        else:
            pseq_dict[allele].append([new_pseq, collator(tokenizer(
                pseq,
                max_length=300,
                padding="max_length",
                truncation=True,
                return_tensors="np"  # 确保返回numpy数组

            )['input_ids'])['input_ids'], collator(tokenizer(
                pseq,
                max_length=300,
                padding="max_length",
                truncation=True,
                return_tensors="np"  # 确保返回numpy数组

            )['input_ids'])['labels']])

    return pseq_dict


def read_binding_data(path, pseq_dict, global_args):
    [blosum_matrix, aa, main_dir, output_path] = global_args
    data_dict = {}
    f = open(path, "r")

    for line in f:
        info = re.split("\t", line)

        allele = info[1]
        if allele in pseq_dict.keys():
            affinity = 1 - log(float(info[5])) / log(50000)
            pep = info[3]  # Sequence of the peptide in the form of a string, like "AAVFPPLEP"
            pep_blosum = []  # Encoded peptide seuqence
            for residue_index in range(12):
                # Encode the peptide sequence in the 1-12 columns, with the N-terminal aligned to the left end
                # If the peptide is shorter than 12 residues, the remaining positions on
                # the rightare filled will zero-padding
                if residue_index < len(pep):
                    pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
                else:
                    pep_blosum.append(np.zeros(20))
            for residue_index in range(12):
                # Encode the peptide sequence in the 13-24 columns, with the C-terminal aligned to the right end
                # If the peptide is shorter than 12 residues, the remaining positions on
                # the left are filled will zero-padding
                if 12 - residue_index > len(pep):
                    pep_blosum.append(np.zeros(20))
                else:
                    pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 12 + residue_index]]])
            # new_data = [encoded pep sequence, encoded MHC pseudo sequence, len of pep, affinity]
            new_data = [pep_blosum, pseq_dict[allele], affinity, len(pep), pep]
            if allele not in data_dict.keys():
                data_dict[allele] = [new_data]
            else:
                data_dict[allele].append(new_data)
    print("Finished reading binding data")
    return data_dict


def read_binding_data_esm(path, pseq_dict, global_args):
    [blosum_matrix, aa, main_dir, output_path] = global_args
    data_dict = {}
    f = open(path, "r")
    new_dir = '/root/ckpt'
    pdb_dir = '/root/adjacency_matrices.txt'
    tokenizer = AutoTokenizer.from_pretrained(new_dir)
    allele_matrices = load_adjacency_matrices(pdb_dir)
    for line in f:
        info = re.split("\t", line)
        allele = info[1]
        if allele in pseq_dict.keys():
            # print('$###################################',allele)
            adj = allele_matrices[allele]
            # print(np.array(adj).shape)
            affinity = 1 - log(float(info[5])) / log(50000)
            pep = info[3]  # Sequence of the peptide in the form of a string, like "AAVFPPLEP"
            pep_blosum = []  # Encoded peptide seuqence
            for residue_index in range(12):
                # Encode the peptide sequence in the 1-12 columns, with the N-terminal aligned to the left end
                # If the peptide is shorter than 12 residues, the remaining positions on
                # the rightare filled will zero-padding
                if residue_index < len(pep):
                    pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
                else:
                    pep_blosum.append(np.zeros(20))
            for residue_index in range(12):
                # Encode the peptide sequence in the 13-24 columns, with the C-terminal aligned to the right end
                # If the peptide is shorter than 12 residues, the remaining positions on
                # the left are filled will zero-padding
                if 12 - residue_index > len(pep):
                    pep_blosum.append(np.zeros(20))
                else:
                    pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 12 + residue_index]]])

            # new_data = [encoded pep sequence, encoded MHC pseudo sequence, len of pep, affinity]
            new_data = [pep_blosum, pseq_dict[allele][0], affinity,adj, len(pep),pseq_dict[allele][1],
                        pseq_dict[allele][2], tokenizer(
                    pep,
                    max_length=12,
                    padding="max_length",
                    truncation=True,
                    return_tensors="np"  # 确保返回numpy数组

                )['input_ids'], pep]
            if allele not in data_dict.keys():
                data_dict[allele] = [new_data]
            else:
                data_dict[allele].append(new_data)
    print("Finished reading binding data")
    del allele_matrices
    return data_dict


def redundancy_removal(data_dict):
    '''
    Removes the redundant data from the training set
    '''
    for allele in sorted(data_dict.keys()):
        allele_data = data_dict[allele]
        unique_9mers = []
        nonredundant_data = []
        overlap = 0
        for pep_info in allele_data:
            redundant = False
            seq = pep_info[-1]
            for i in range(len(seq) - 9 + 1):
                _9mer = seq[i:i + 9]
                if _9mer not in unique_9mers:
                    unique_9mers.append(_9mer)
                else:
                    redundant = True
            if not redundant or len(seq) == 8:
                nonredundant_data.append(pep_info)
        data_dict[allele] = nonredundant_data

    return data_dict


def read_validation_data_esm(path, pseq_dict, global_args):
    [blosum_matrix, aa, main_dir, output_path] = global_args
    data_dict = {}
    f = open(path, "r")
    new_dir = '/root/ckpt'
    pdb_dir = '/root/adjacency_matrices.txt'
    tokenizer = AutoTokenizer.from_pretrained(new_dir)
    allele_matrices = load_adjacency_matrices(pdb_dir)
    for line in f:
        info = re.split("\t", line)
        allele = info[1]
        if allele in pseq_dict.keys():
            # print('$###################################',allele)
            adj = allele_matrices[allele]
            # print(np.array(adj).shape)
            affinity = 1 - log(float(info[5])) / log(50000)
            pep = info[3]  # Sequence of the peptide in the form of a string, like "AAVFPPLEP"
            pep_blosum = []  # Encoded peptide seuqence
            for residue_index in range(12):
                # Encode the peptide sequence in the 1-12 columns, with the N-terminal aligned to the left end
                # If the peptide is shorter than 12 residues, the remaining positions on
                # the rightare filled will zero-padding
                if residue_index < len(pep):
                    pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
                else:
                    pep_blosum.append(np.zeros(20))
            for residue_index in range(12):
                # Encode the peptide sequence in the 13-24 columns, with the C-terminal aligned to the right end
                # If the peptide is shorter than 12 residues, the remaining positions on
                # the left are filled will zero-padding
                if 12 - residue_index > len(pep):
                    pep_blosum.append(np.zeros(20))
                else:
                    pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 12 + residue_index]]])

            # new_data = [encoded pep sequence, encoded MHC pseudo sequence, len of pep, affinity]
            new_data = [pep_blosum, pseq_dict[allele][0], affinity, adj, len(pep), pseq_dict[allele][1],
                        pseq_dict[allele][2], tokenizer(
                    pep,
                    max_length=12,
                    padding="max_length",
                    truncation=True,
                    return_tensors="np"  # 确保返回numpy数组

                )['input_ids'], pep]
            if allele not in data_dict.keys():
                data_dict[allele] = [new_data]
            else:
                data_dict[allele].append(new_data)
    print("Finished reading binding data")
    return data_dict


def preparing_data(data_dict, cross_validation, n_splits, s, test_len=9):
    # training_data = []
    # test_dicts = []
    # cross_validation = KFold(n_splits = n_splits)
    # cross_validation = KFold(n_splits=n_splits)

    split_indices_train = []
    split_indices_test = []

    # For each partition, initialize the container of data and target
    for split in range(n_splits):
        # training_data.append([])
        # test_dicts.append([])
        split_indices_test.append([])
        split_indices_train.append([])
    for allele in data_dict.keys():
        allele_data = data_dict[allele]
        # random.shuffle(allele_data)
        # print(allele_data)
        # allele_data = np.array(allele_data)
        split = 0
        # We only include the alleles with >= data
        if len(allele_data) < 100:
            continue
        # Partition of data
        for training_indices, test_indices in cross_validation.split(allele_data):
            split_indices_train[split].extend([training_indices])
            split_indices_test[split].extend([test_indices])

            # print(len(allele_data),len(allele_data[0]))
            # training_data[split].extend([allele_data[i] for i in training_indices])
            # test_dicts[split].extend([allele_data[i] for i in test_indices])
            # print()
            split += 1
    print(len(split_indices_train[0]))
    # for split in range(n_splits):
    #     random.shuffle(training_data[split])
    return [split_indices_train[s], split_indices_test[s]]


seq1_len = 15  # 肽段长度
seq2_len = 30  # MHC长度
adj_size = 324
batch_size = 32
epochs = 200
main_dir = "/root/"
path_train = main_dir + "data/binding_data_train.txt"
output_path = "/root/autodl-tmp/result/"
aa = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9, "L": 10, "K": 11, "M": 12,
      "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}
# Load the blosum matrix for encoding
path_blosum = main_dir + r"blosum50.txt"
blosum_matrix = read_blosum(path_blosum)
# 创建模型


# 模拟输入
path_seq = main_dir + "HLA_all.txt"
seq_dict = allele_seq(path_seq)
print(len(seq_dict))
global_args = [blosum_matrix, aa, main_dir, output_path]
pseq_dict = pseudo_seq_esm(seq_dict, global_args)
# data_dict = read_binding_data_esm(path_train, pseq_dict, global_args)
# # print(len(data_dict))
# data_dict = redundancy_removal(data_dict)
path_val = main_dir + "data/binding_data_val.txt"
# 测试集
validation_data = read_validation_data_esm(path_val, pseq_dict, global_args)
validation_data = redundancy_removal(validation_data)

# Data partition for cross-validation
n_splits = 5
# 交叉验证划分
# training_data, test_dicts = preparing_data(data_dict, n_splits,test_len=9)
# print ("Finished data loading")
# print ("shape of training data", len(training_data),len(training_data[0])  )
cross_validation = KFold(n_splits=n_splits)
# Cross-validation
models = []
for allele in validation_data.keys():
    foutput(allele,output_path + 'result_val.txt')
    allele_data = validation_data[allele]
    cross_validation_test(allele_data, allele_data,
                          global_args)
# for split in range(n_splits):
#     training_data = []
#     test_dicts = []
#     # training_data, test_dicts = preparing_data(data_dict, cross_validation, split, test_len=9)
#     [training_indices, test_indices] = preparing_data(data_dict, cross_validation, n_splits, split)
#     l = 0
#     for allele in data_dict.keys():
#         allele_data = data_dict[allele]
#         if len(allele_data) < 100:
#             continue
#         # Partition of data
#         # print(len(allele_data),len(allele_data[0]),allele_data)
#         # print(training_indices[l])
#         training_data.extend([allele_data[i] for i in training_indices[l]])
#         test_dicts.extend([allele_data[i] for i in test_indices[l]])
#         l += 1
#     random.shuffle(training_data)
#     random.shuffle(test_dicts)
#     print("Finished data loading, fold is:", split)
#     print("shape of training data", len(test_dicts), len(test_dicts[0]))
#
#     cross_validation_test(training_data, test_dicts,
#                                                              global_args)
#     # performance_dicts.append(performance_dict)

