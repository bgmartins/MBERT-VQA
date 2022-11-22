'''
    code from: https://github.com/sarthmit/Compositional-Attention/blob/main/Multitask_Classification/model.py

    @misc{mittal2021compositional,
      title={Compositional Attention: Disentangling Search and Retrieval}, 
      author={Sarthak Mittal and Sharath Chandra Raparthy and Irina Rish and Yoshua Bengio and Guillaume Lajoie},
      year={2021},
      eprint={2110.09419},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
    
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Compositional_Attention(nn.Module):
    def __init__(self, dim, qk_dim=16, nheads=4, nrules=1, dot=False):
        super(Compositional_Attention, self).__init__()

        self.dim = dim
        self.nheads = nheads
        self.nrules = nrules
        self.head_dim = dim // nheads
        self.qk_dim = qk_dim
        self.dot = dot

        self.norm_before = True

        self.query_net = nn.Linear(dim, dim)
        self.key_net = nn.Linear(dim, dim)
        self.value_net = nn.Linear(dim, self.head_dim * self.nrules)

        self.query_value_net = nn.Linear(dim, self.qk_dim * nheads)

        if dot:
            self.key_value_net = nn.Linear(self.head_dim, self.qk_dim)
        else:
            self.score_network = nn.Linear(self.head_dim + self.qk_dim, 1)

        self.final = nn.Linear(dim, dim)

        self.res = nn.Sequential(
            nn.Linear(dim,2 * dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.Dropout(p=0.1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, vis=False, mask=None, prev=None):
        bsz, n_read, _ = x.shape
        _, n_write, _ = x.shape

        res = x
        if self.norm_before:
            x = self.norm1(x)

        q = self.query_net(x).reshape(bsz, n_read, self.nheads, self.head_dim)
        q = q.permute(0,2,1,3) / np.sqrt(self.head_dim)
        k = self.key_net(x).reshape(bsz, n_write, self.nheads, self.head_dim)
        k = k.permute(0,2,3,1)
        v = self.value_net(x).reshape(bsz, n_write, self.nrules, self.head_dim)
        v = v.permute(0,2,1,3).unsqueeze(1)

        score = torch.matmul(q,k)

        if mask is not None: # mask in 1st att mechanism
            #mask = mask.unsqueeze(-1).unsqueeze(-1).permute(0,2,3,1)
            
            #mask = mask.expand(score.size()).float()
            mask = mask[:, None, None, :].float() #[B,1,1,n_read]
            score -= 10000.0 * (1.0 - mask)

        score = F.softmax(score, dim=-1).unsqueeze(2) # (bsz, nheads, n_read, n_write)

        out = torch.matmul(score, v) # (bsz, nheads, nrules, n_read, att_dim)
        out = out.view(bsz, self.nheads, self.nrules, n_read, self.head_dim)

        out = out.permute(0, 3, 1, 2, 4).reshape(bsz, n_read, self.nheads, self.nrules, self.head_dim)

        if self.dot:
            q_v = self.query_value_net(x).reshape(bsz, n_read, self.nheads, 1, self.qk_dim) / np.sqrt(self.qk_dim)
            k_v = self.key_value_net(out).reshape(bsz, n_read, self.nheads, self.nrules, self.qk_dim)
            if prev is not None:
                comp_score = torch.matmul(q_v, k_v.transpose(4,3)) + prev #residual connections between attention like in RealFormer
            else:
                comp_score = torch.matmul(q_v, k_v.transpose(4,3))
        else:
            q_v = self.query_value_net(x).reshape(bsz, n_read, self.nheads, 1, self.qk_dim).expand(-1, -1, -1, self.nrules, -1)
            in_ = torch.cat((q_v, out), dim=-1)
            if prev is not None:
                comp_score = self.score_network(in_) + prev
            else:
                comp_score = self.score_network(in_)
        
        prev = comp_score

        comp_score = comp_score.reshape(bsz, n_read, self.nheads, self.nrules, 1)
        

        if mask is not None: # mask in 2nd att mechanism
            
            mask = mask.unsqueeze(-1).float() #this mask has one more dim than the other - the 1
            
            mask = mask.reshape(bsz, n_read, 1, 1, 1) # had to this reshape
            
            #mask = mask.unsqueeze(-1).unsqueeze(-1).expand(comp_score.size()).float()
            comp_score -= 10000.0 * (1.0 - mask)

        

        comp_score = F.softmax(comp_score, dim=3)

        out = (comp_score * out).sum(dim=3).reshape(bsz, n_read, self.dim)

        out = self.final(out)

        if not self.norm_before:
            out = self.norm1(res + out)
        else:
            out = res + out

        res = out

        if self.norm_before:
            out = self.norm2(out)
            out = res + self.res(out)
        else:
            out = self.norm2(res + self.res(out))

        if vis:
            return out, comp_score

        return out, prev