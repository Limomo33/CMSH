# 在模型定义部分新增以下内容
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import log
import os
from torch.utils.data import DataLoader, TensorDataset
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

import json
class ESMModelWrapper(nn.Module):
    def __init__(self, esm_model_name="esm2_t6_8M_UR50D"):
        super().__init__()
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(esm_model_name)
        #self.batch_converter = self.alphabet.get_batch_converter()
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
        # sequences = sequences.squeeze().to(device)
        #print(sequences.size())
        #batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        with torch.no_grad():
            results = self.esm_model(sequences, repr_layers=[self.repr_layer])
        token_embeddings = results["representations"][self.repr_layer]
        # print(token_embeddings.size())
        contrast_emb = self.contrast_proj(token_embeddings.mean(1))

        #embeddings = self._remove_special_tokens(token_embeddings, sequences)
        return token_embeddings, contrast_emb
        
class EnhancedESMModelWrapper(nn.Module):
    def __init__(self, esm_model_name="esm2_t6_8M_UR50D"):
        super().__init__()
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(esm_model_name)
        
        self._add_lora_adapters()
        
        self.contrast_proj = nn.Sequential(
            nn.Linear(self.esm_model.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def _add_lora_adapters(self):
        lora_config = LoraConfig( use_dora=True,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.5,
            bias="none"
        )
        self.esm_model = get_peft_model(self.esm_model, lora_config)

    def forward(self, sequences):
        results = self.esm_model(sequences, repr_layers=[self.esm_model.num_layers])
        #print(results)
        token_embeddings = results["representations"][self.esm_model.num_layers]
        contrast_emb = self.contrast_proj(token_embeddings.mean(1))
        lables=results["logits"]
        # print('111111111111111',lables.size(),token_embeddings.size())
        return token_embeddings, contrast_emb, lables
        
def compute_loss(logits, labels):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, batch_size=256, num_head=2,dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_head=num_head
        self.W = nn.Parameter(torch.zeros(size=( in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.Adj = nn.Parameter(torch.ones(size=(batch_size,324, 324)))
        nn.init.xavier_uniform_(self.Adj.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):

        batch_size, N, _ = x.size()
        h = torch.matmul(x, self.W)

        #N = x.size()[0]
        #h_flat = h.view(batch_size, -1)
        #print('111111111111', adj.size())
        #a_input = torch.cat([h.repeat(1, 1,N).view(N * N, self.out_features), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)

        h=F.dropout(h, self.dropout, training=self.training)
        Wh1 = torch.matmul(h, self.a[:self.out_features, :])
        Wh2 = torch.matmul(h, self.a[self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        #print('111111111111', Wh1.size(),Wh2.size())

        # a_input = torch.cat([h.repeat(1, seq_len).view(N, seq_len * seq_len, -1),
        #                      h.repeat(1, seq_len, 1)], dim=-1)
        # a_input = a_input.view(N, seq_len, seq_len, 2 * self.out_features)

        attention = self.leakyrelu(e)
        #print('111111', attention.size())

        #zero_vec = -10e9 * torch.ones_like(attention)
        #mask=zero_vec* self.Adj
        #attention = torch.where(adj > 0, attention, zwero_vec)
        attention=attention*self.Adj[:attention.size()[0],:,:]
        #print('22222222', attention.size())

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        #print('111111111111', h_prime.size())

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class PositionEmbeddingTrain(nn.Module):
    def __init__(self, max_position, dim):
        super().__init__()
        self.pos_embed = nn.Embedding(max_position, dim)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        positions = self.pos_embed(positions)
        return torch.cat((x,positions),dim=2)

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


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, n_final_out, dropout=0.5, alpha=0.2, nheads=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, n_final_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        #print('11111111111',x.size())

        return x


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
    def __init__(self, seq1_len, seq2_len, dim=340, kernel_size=9, num_heads=2, gat_heads=8, fc2_size=1):
        super().__init__()
        self.pos_embed1 = PositionEmbeddingTrain(seq1_len, 20)
        self.pos_embed2 = PositionEmbeddingTrain(seq2_len, 20)

        self.attention1 = MultiHeadAttention(340)
        self.attention2 = MultiHeadAttention(340)

        self.gat = GAT(nfeat=dim, nhid=64,n_final_out=32)  #
        self.linear1 = nn.Linear(2720, 256)  # 
        self.linear2 = nn.Linear(3200, 256)  # 
        self.linear3 = nn.Linear(10720, 256)  # 
        self.linear4 = nn.Linear(512, 64)#11136
        self.linear5 = nn.Linear(64, 1)

        self.conv1 = nn.Conv1d(
            in_channels=300,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding='same'
        )
        self.convp = nn.Conv1d(
            in_channels=24,
            out_channels=8,
            kernel_size=kernel_size,
            stride=1,
            padding='same'
        )
        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=32,
            kernel_size=kernel_size,
            stride=1,
            padding='same'
        )
        self.flatten = nn.Flatten()  # 

        self.pool = nn.AvgPool1d(kernel_size=1)
        self.dropout = nn.Dropout(0.5)
        #self.linear=nn.Linear(out_features=2)
        self.norm1=nn.LayerNorm(340, eps=1e-5, elementwise_affine=True)
        self.norm2=nn.LayerNorm(1, eps=1e-5, elementwise_affine=True)

        # 
        self.esm_encoder = EnhancedESMModelWrapper()
        self.esm_encoder2 = ESMModelWrapper()
        self.fixed_noise = TrainableGaussianNoise(
    initial_std=0.2,
    trainable=False,
    always_active=True,
    clamp_output=(-3, 3))

        # 
        self.contrast_loss = nn.L1Loss(reduction='mean')
        # self.contrast_loss=nn.CrossEntropyLoss()
        self.contrast_weight =0.3  # 

        self.com_loss= nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, pep, mhc, adj, mhc_seq, pep_seq):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 
        pep_embeddings, pep_contrast = self.esm_encoder2(pep_seq)
        mhc_embeddings, mhc_contrast,l = self.esm_encoder(mhc_seq)
        
        pep_embeddings = torch.cat([pep_embeddings, pep_embeddings],dim=1)  # 
        # mhc_embeddings = self._pad_sequences(mhc_embeddings, target_len=300)  # 
        pepesm = self.pos_embed1(pep_embeddings)
        mhcesm = self.pos_embed2(mhc_embeddings)
        pepesm=self.fixed_noise(pepesm)     
        mhcesm=self.dropout(mhcesm)

        #print(pep.size(),mhc.size())

        pep_attn = self.attention1(pepesm)
        mhc_attn = self.attention2(mhcesm)#(batch,seq,dim)
        #print('11111',mhc_attn.size())(batch,seq,dim)
        pep_pool = self.pool(pep_attn)
        mhc_pool = self.pool(mhc_attn)
        # print('11111',mhc_pool.size())
        mhc_att=self.dropout(mhc_pool)
        mhc_att=self.norm1(mhc_att)
        pep_pool=self.norm1(pep_pool)

        # combined = torch.cat([pep, mhc], dim=1)
        mhc=self.fixed_noise(mhc)    #gat_out = self.gat(mhcbined, adj)
        #Y=self.flatten(gat_out)

        mhc=self.conv1(mhc)
        mhc_conv=self.conv2(mhc)
        mhc_conv=self.dropout(mhc)


        pep_conv=self.convp(pep)
        
        flat_pep_0=self.flatten(pep_conv)
        flat_pep_1=self.flatten(pep_pool)
        flat_mhc_0 = self.flatten(mhc_att)  #
        flat_mhc_1 = self.flatten(mhc)
        flat_mhc_2 = self.flatten(mhc_conv)

        # cat_0=torch.cat([flat_pep_0,flat_mhc_0],dim=1)

        cat_1 = torch.cat([flat_pep_0, flat_mhc_1],dim=1)

        cat_2 = torch.cat([flat_pep_1, flat_mhc_2],dim=1)

        fc1_0 = self.linear1(cat_1)
        fc1_0=F.relu(fc1_0)
        # fc1_1 = self.linear2(cat_1)
        # fc1_1=F.relu(fc1_1)
        fc1_2 = self.linear3(cat_2)
        fc1_2=F.relu(fc1_2)
        merge_1 = torch.cat([fc1_0, fc1_2],dim=1)
        fc2 = self.linear4(merge_1)  # merge_1
        fc2=F.relu(fc2)  
        # fc2=self.norm1(fc2)
        out = self.linear5(fc2)
        out=F.leaky_relu(out, negative_slope=0.002)
        return out, pep_attn.mean(1), mhc_att.mean(1),l

# 
def cross_validation_training_transformer_gat(training_data, test_dict, validation_data, validation_target,global_args):

    [blosum_matrix, aa, main_dir, output_path] = global_args
    training_pep, training_mhc, training_target = [[i[j] for i in training_data] for j in range(3)]
    training_mhc_seq, training_mhc_l,training_pep_seq=[[i[j] for i in training_data] for j in range(4,7)]
    validation_pep, validation_mhc, validation_target = [[i[j] for i in test_dict] for j in range(3)]
    validation_mhc_seq,validation_mhc_l, validation_pep_seq=[[i[j] for i in test_dict] for j in range(4,7)]
    #print(training_mhc_seq)
    #validation_pep, validation_mhc = [i[0] for i in validation_data], [i[1] for i in validation_data]

    training_pep = torch.FloatTensor(np.array(training_pep))
    training_mhc = torch.FloatTensor(np.array(training_mhc))
    training_target = torch.FloatTensor(np.array(training_target))
    training_mhc_l = torch.LongTensor(np.array(training_mhc_l))

    training_pep_seq = torch.LongTensor(np.array(training_pep_seq).squeeze())
    training_mhc_seq = torch.LongTensor(np.array(training_mhc_seq).squeeze())

    validation_pep = torch.FloatTensor(np.array(validation_pep))
    validation_mhc = torch.FloatTensor(np.array(validation_mhc))
    validation_pep_seq = torch.LongTensor(np.array(validation_pep_seq).squeeze())
    validation_mhc_seq = torch.LongTensor(np.array(validation_mhc_seq).squeeze())
    validation_target = torch.FloatTensor(np.array(validation_target))
    validation_mhc_l = torch.LongTensor(np.array(validation_mhc_l))

    # print(training_mhc_l.size(),validation_mhc_l.size(),validation_target.size())
    aa = creat_adj()  # 
    aa = np.expand_dims(aa, axis=0).astype(np.float32)
    A = np.tile(aa, (len(training_mhc), 1, 1))
    val_A=np.tile(aa, (len(validation_mhc), 1, 1))
    A = torch.FloatTensor(A)
    val_A= torch.FloatTensor(val_A)


    fc2_size = 1
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs=150
    # 

    model = TransformerModel(
        seq1_len=training_pep.shape[1],
        seq2_len=training_mhc.shape[1],
        fc2_size=fc2_size
    ).to(device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.00002}
        #{'params': model.esm_encoder.esm_model.parameters(), 'lr': 1e-5}
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
            # 
            dataset = TensorDataset(training_pep, training_mhc, A, training_target,training_mhc_seq,training_pep_seq,training_mhc_l)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            total_loss = 0
            t_loss=0

            # 
            for batch_pep, batch_mhc, batch_adj, batch_target, batch_mhc_seq, batch_pep_seq,batch_mhc_l in tqdm(loader,desc=f'epoch{epoch+1}/{num_epochs}'):
                batch_pep = batch_pep.to(device)
                batch_mhc = batch_mhc.to(device)
                batch_adj = batch_adj.to(device)
                batch_target = batch_target.to(device)
                batch_mhc_l=batch_mhc_l.to(device)

                batch_mhc_seq = batch_mhc_seq.to(device)
                batch_pep_seq = batch_pep_seq.to(device)
                # print(batch_mhc_seq.size(),batch_pep_seq.size())
                optimizer.zero_grad()

                outputs, pep_contrast, mhc_contrast,l = model(batch_pep, batch_mhc, batch_adj,batch_mhc_seq,batch_pep_seq)

                # 
                shuffle_idx = torch.randperm(pep_contrast.size(0))
                neg_contrast = pep_contrast[shuffle_idx]
                cont_loss = model.contrast_loss(
                    pep_contrast, 
                    mhc_contrast  # 
                    # neg_contrast    # 
                )
                # cont_loss=simcse_unsup_loss(pep_contrast,mhc_contrast)
                # mlm_loss = compute_loss(mhc_l, batch_mhc_l)
                # 
                total_loss = criterion(outputs.squeeze(), batch_target) +model.contrast_weight * cont_loss+0.1*model.com_loss(l.view(-1, l.size(-1)), batch_mhc_l.view(-1))
                t_loss += total_loss.item()
  
                total_loss.backward()
                optimizer.step()
          

            train_loss=t_loss/len(loader)
  # 
            print(train_loss)
            
            with torch.no_grad():
                v_loss =0

                # 
                model.eval()
                valdataset = TensorDataset(validation_pep, validation_mhc, val_A, validation_target, validation_mhc_seq,validation_pep_seq,validation_mhc_l)
                val_loader = DataLoader(valdataset, batch_size=256, shuffle=True)
    
                for batch_vpep, batch_vmhc, batch_vdj, batch_vtarget,batch_vmhc_seq, batch_vpep_seq,batch_vmhc_l in val_loader:
                    batch_vpep = batch_vpep.to(device)
                    batch_vmhc = batch_vmhc.to(device)
                    batch_vdj = batch_vdj.to(device)
                    batch_vtarget = batch_vtarget.to(device)
                    batch_vmhc_l=batch_vmhc_l.to(device)
                    batch_vmhc_seq = batch_vmhc_seq.to(device)
                    batch_vpep_seq = batch_vpep_seq.to(device)
                    #optimizer.zero_grad()
                    outputs,pep_contrast,mhc_contrast,l = model(batch_vpep, batch_vmhc, batch_vdj,batch_vmhc_seq,batch_vpep_seq)
                    # print('outyput',outputs)
    
                    # shuffle_idx = torch.randperm(pep_contrast.size(0))
                    # neg_contrast = pep_contrast[shuffle_idx]
                    # cont_loss = model.contrast_loss(
                    #     pep_contrast, 
                    #     mhc_contrast,  # 
                    #     neg_contrast    # 
                    # )
                    # mlm_loss = compute_loss(mhc_l, batch_mhc_l)
   
                    # 
                    total_loss = criterion(outputs.squeeze(), batch_vtarget)              
                    pcc, roc_auc, max_acc = model_eval(outputs.squeeze(), batch_vtarget)
                    #loss.backward()
                    #optimizer.step()
                    val_pcc+=pcc
                    val_roc+=roc_auc
                    val_acc+=max_acc
                    v_loss += total_loss.item()
    
                val_loss = v_loss / len(val_loader)
                val_pcc=val_pcc/len(val_loader)
                val_roc=val_roc/len(val_loader)
                val_acc=val_acc/len(val_loader)


                poor_init = True

                # if epoch == 0 and not ((val_pcc< 1) and (val_pcc>0.1)):
                #     poor_init = True
                #     break
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f},Test pcc: {val_pcc:.4f},Test roc: {val_roc:.4f},Test acc: {val_acc:.4f}")
                foutput(str(epoch)+"\t"+str(train_loss)+"\t"+str(val_loss)+"\t"+str(val_pcc)+"\t"+str(val_roc)+"\t"+str(val_acc),output_path+'result.txt')
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
    
            plt.savefig('loss_gsm_esm.png')  #
  # if poor_init:
  #               continue
  #           if val_pcc > 0.8:
  #               torch.save(model, 'gsm_esm.pt')
  #               models.append(model)

    print("Training complete.")
    #performance_dict = model_performance(models, validation_data, validation_target, aa, global_args)

    return models
def model_eval(predictions, targets, threshold=0.5):
    #from torchmetrics import PearsonCorrCoef

    # if isinstance(predictions, torch.Tensor):
    pred_np = predictions.cpu().detach().numpy()
    target_np = targets.cpu().detach().numpy()
    print(pred_np)
    # bin_targets = (targets > threshold).astype(int)
    # bin_preds = (predictions > threshold).astype(int)
    #
    # try:
    pcc,_ = pearsonr(pred_np, target_np)
    # except:
    #     pcc = 0.0

    #try:
    test_label = [1 if aff > 1 - log(500) / log(50000) else 0 for aff in target_np]
    fpr, tpr, thresholds = roc_curve(test_label, pred_np)
    roc_auc = auc(fpr, tpr)
    # except:
    #     roc_auc = 0.5

    threshold = 1 - log(500) / log(50000)
    p = [0 if score < threshold else 1 for score in pred_np]
    accurate = [1 if p[i] == test_label[i] else 0 for i in range(len(p))]
    acc = np.sum(accurate) / float(len(accurate))


    return pcc,roc_auc,acc


def model_performance(models, validation_data, validation_target, aa, global_args):

    [blosum_matrix, _, main_dir, output_path] = global_args
    test_pep, test_mhc= [[i[j] for i in validation_data] for j in range(2)]
    #
    # test_pep = np.array([i[0] for i in test_dict["data"]])
    # test_mhc = np.array([i[1] for i in test_dict["data"]])
    #test_target = np.array(test_dict["target"])
    test_pep = torch.FloatTensor(np.array(test_pep))
    test_mhc = torch.FloatTensor(np.array(test_mhc))
    test_target = torch.FloatTensor(np.array(validation_target)).unsqueeze(1)

    # 
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

            pcc, roc_auc, max_acc = model_eval(outputs, test_target)

            # 
            performance_dict[f"model_{i + 1}"] = {
                "PCC": pcc,
                "AUC": roc_auc,
                "Max_Accuracy": max_acc,
                "Predictions": outputs.tolist(),
                "Targets": test_target.tolist()
            }

            print(f"Model {i + 1} Performance:")
            print(f"PCC: {pcc:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Max Accuracy: {max_acc:.4f}")
            print("-" * 40)

    return performance_dict# 
def model_eval(predictions, targets, threshold=0.5):
    #from torchmetrics import PearsonCorrCoef

    # if isinstance(predictions, torch.Tensor):
    pred_np = predictions.cpu().detach().numpy()
    target_np = targets.cpu().detach().numpy()

    # bin_targets = (targets > threshold).astype(int)
    # bin_preds = (predictions > threshold).astype(int)
    #
    # # 
    # try:
    pcc,_ = pearsonr(pred_np, target_np)
    # except:
    #     pcc = 0.0

    #try:
    test_label = [1 if aff > 1 - log(500) / log(50000) else 0 for aff in target_np]
    fpr, tpr, thresholds = roc_curve(test_label, pred_np)
    roc_auc = auc(fpr, tpr)
    # except:
    #     roc_auc = 0.5

    threshold = 1 - log(500) / log(50000)
    p = [0 if score < threshold else 1 for score in pred_np]
    accurate = [1 if p[i] == test_label[i] else 0 for i in range(len(p))]
    acc = np.sum(accurate) / float(len(accurate))


    return pcc,roc_auc,acc


def model_performance(models, validation_data, validation_target, aa, global_args):

    [blosum_matrix, _, main_dir, output_path] = global_args
    test_pep, test_mhc= [[i[j] for i in validation_data] for j in range(2)]
    #
    # test_pep = np.array([i[0] for i in test_dict["data"]])
    # test_mhc = np.array([i[1] for i in test_dict["data"]])
    #test_target = np.array(test_dict["target"])
    test_pep = torch.FloatTensor(np.array(test_pep))
    test_mhc = torch.FloatTensor(np.array(test_mhc))
    test_target = torch.FloatTensor(np.array(validation_target)).unsqueeze(1)

    # 
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
        plt.figure(figsize=(10,8))
        plt.scatter(contrast_emb[:,0], contrast_emb[:,1], c=labels)
        plt.colorbar()
        plt.title(f"Contrastive Embeddings (Epoch {epoch})")
        plt.savefig(f"contrast_epoch{epoch}.png")
        plt.close()
