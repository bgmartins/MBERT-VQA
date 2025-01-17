from models.transformer import BertLayer
from models.feedback_transformer_pytorch import FeedbackTransformer
from models.realformer import ResEncoderBlock
from models.comp_attention import Compositional_Attention
from models.image_encoding import get_transfer
from models.serf import SERF
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision import models
import math
import numpy as np
import torch.nn.functional as F
import timm
import copy


def get_bert_model(args):
    if args.task == 'distillation':
            bert_name = args.clinicalbert
    else:
        bert_name = 'bert-base-uncased'
    return bert_name

def get_transformer_model(args):
    if 'feedback-transformer' in args.transformer_model:
        print('Using Feedback Transformer')
        #x = torch.randint(0, 20000, (2, 64))
        #x = torch.rand(3,64,768)
        return FeedBackTransformer(args)
    elif 'realformer' in args.transformer_model:
        print('Using RealFormer')
        return RealFormer(args)
    elif 'transformer' in args.transformer_model:
        print('Using regular transformer')
        return Transformer(args)
    elif 'compositional' in args.transformer_model:
        print('Using Compositional Attention')
        return Compositional_Transformer(args)
    else:
        raise NotImplementedError


class TransformerAbstract(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bert_embedding = self.get_bert_embedding(args)
        self.trans = get_transfer(args)

    def get_bert_embedding(self,args):
        bert_name = get_bert_model(args)
        base_model = AutoModel.from_pretrained(bert_name)
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        return bert_model[0]

    # encode images with cnn and embedd the text tokens and prepare
    # to feed to the transformer
    def prepare_input(self, img, input_ids, token_type_ids, mask):
        # turn into a list to iterate an arbitrary number of tokens
        vizs = list(self.trans(img)) 
        h = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)
        for n, v in enumerate(vizs):
            for i in range(len(h)):
                h[i][n] = v[i]
        return h
        

class Transformer(TransformerAbstract):
    def __init__(self, args):
        super().__init__(args)
        print('Transformer from abstract')
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers

    def forward(self, img, input_ids, token_type_ids, mask):
        h = self.prepare_input(img, input_ids, token_type_ids, mask)
        for i in range(self.n_layers):
            h = self.blocks(h, mask, i)
        return h

class RealFormer(TransformerAbstract):
    def __init__(self, args):
        super().__init__(args)
        #i used head_cnt = 8 in RealFormer
        head_cnt = 8
        print('RealFormer from abstract, heads', head_cnt)
        self.mains = nn.Sequential(*[ResEncoderBlock(emb_s = args.hidden_size // head_cnt, head_cnt = head_cnt, dp1 = 0.1, dp2 = 0.1) for _ in range(args.n_layers)])
    def forward(self, img, input_ids, token_type_ids, mask):
        h = self.prepare_input(img, input_ids, token_type_ids, mask)
        prev = None
        for resencoder in self.mains:
            h, prev = resencoder(h, prev = prev, mask = mask)
        return h

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Compositional_Transformer(TransformerAbstract):
    def __init__(self, args):
        super().__init__(args)
        #i used head_cnt = 8 in RealFormer
        head_cnt = 8
        print('Compositional Attention from abstract, with mask and residuals heads', head_cnt)
        self.n_layers = args.n_layers
        layer = Compositional_Attention(dim = args.hidden_size, qk_dim= args.hidden_size // head_cnt, nheads=head_cnt, nrules=head_cnt, dot=True) #dot=True for compositional att
        self.layers = _get_clones(layer, self.n_layers)
    def forward(self, img, input_ids, token_type_ids, mask):
        h = self.prepare_input(img, input_ids, token_type_ids, mask)
        prev = None
        for module in self.layers:
            h, prev = module(h, mask=mask, prev=prev)
        return h

class FeedBackTransformer(TransformerAbstract):
    def __init__(self, args):
        super().__init__(args)
        self.block = FeedbackTransformer(
            num_tokens = args.vocab_size,           # number of tokens
            dim = args.hidden_size,                    # dimension
            depth = args.n_layers,                    # depth
            seq_len = 2,                  # the sequence length of each segment or window
            mem_len = 256,                # length of the memory buffer
            dim_head = 64,                # dimension of each head
            heads = 8,                    # number of heads
            attn_dropout = 0.1,           # attention dropout
            ff_dropout = 0.1              # feedforward dropout
        )

    def forward(self, img, input_ids, token_type_ids, mask):
        h = self.prepare_input(img, input_ids, token_type_ids, mask)
        return self.block(h)

class Model(nn.Module):
    def __init__(self,args, feat_dim=128):
        super(Model,self).__init__()
        self.transformer = get_transformer_model(args)
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = SERF()#nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))
        self.task = args.task
        self.dataset = args.dataset

        self.supcon = args.supcon if hasattr(args, 'supcon') else False
        print('supcon task in model', self.supcon)
        if self.supcon:
            self.head = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                SERF(),
                nn.Linear(args.hidden_size, feat_dim)
            )

    def forward(self, img, input_ids, segment_ids, input_mask):
        if self.dataset == 'roco':
            h = self.transformer(img, input_ids, segment_ids, input_mask)
            if self.task == 'MLM':
                pooled_h = self.activ1(self.fc1(h))
                logits = self.classifier(pooled_h)
                if self.supcon: #if supcon, also return the features for the supcon loss
                    feat = F.normalize(self.head(h.mean(1)), dim=1) #reduce dimensions of the features and normalize
                    return logits, feat
            elif self.task == 'distillation':
                logits = h
            return logits

        elif self.dataset == 'VQA-Med':
            h = self.transformer(img, input_ids, segment_ids, input_mask)
            pooled_h = self.activ1(self.fc1(mean_pooling(h, input_mask)))
            logits = self.classifier(pooled_h)
            return logits, 0,0


def mean_pooling(token_embeddings, attention_mask):
    # this is for an huggingface model -> token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)