import os
import numpy as np
import pandas as pd
import random
import math
from PIL import Image
import torch
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from PIL import Image
from random import choice
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_df(file_path):
    paths = os.listdir(file_path)
    df_list = []
    for p in paths:
        df = pd.read_csv(os.path.join(file_path, p), sep='|', names = ['img_id', 'question', 'answer'])
        df['category'] = p.split('_')[1]
        df['mode'] = p.split('_')[2][:-4]
        df_list.append(df)
    return pd.concat(df_list)

def load_data(args, remove = None):

    traindf = pd.read_csv(os.path.join(args.data_dir, 'traindf.csv'))
    valdf = pd.read_csv(os.path.join(args.data_dir, 'valdf.csv'))
    testdf = pd.read_csv(os.path.join(args.data_dir, 'testdf.csv'))

    if remove is not None:
        traindf = traindf[~traindf['img_id'].isin(remove)].reset_index(drop=True)

    # resolution = args.data_dir.split('/')[-1].split('_')[-1]
    print('res',args.data_dir)
    traindf['img_id'] = traindf['img_id'].apply(lambda x: os.path.join(args.data_dir, f'Images_Resized', str(x) + '.jpg'))
    valdf['img_id'] = valdf['img_id'].apply(lambda x: os.path.join(args.data_dir, f'Images_Resized', str(x) + '.jpg'))
    testdf['img_id'] = testdf['img_id'].apply(lambda x: os.path.join(args.data_dir, f'Images_Resized', str(x) + '.jpg'))
    # testdf['img_id'] = testdf['img_id'].apply(lambda x: os.path.join(args.data_dir, x + '.jpg'))

    traindf['category'] = traindf['category'].str.lower()
    valdf['category'] = valdf['category'].str.lower()
    testdf['category'] = testdf['category'].str.lower()


    traindf['answer'] = traindf['answer'].str.lower()
    valdf['answer'] = valdf['answer'].str.lower()
    testdf['answer'] = testdf['answer'].str.lower()

    traindf = traindf.sample(frac = args.train_pct)
    valdf = valdf.sample(frac = args.valid_pct)
    testdf = testdf.sample(frac = args.test_pct)


    return traindf, valdf, testdf

#Utils
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def encode_text(caption,tokenizer, args):
    part1 = [0 for _ in range(5)]
    #get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)[1:-1]

    tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + part2[:args.max_position_embeddings-8] + [tokenizer.sep_token_id]
    segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:args.max_position_embeddings-8])+1)
    input_mask = [1]*len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)

    return tokens, segment_ids, input_mask

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float() ####
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            #print(loss)

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

def crop(img):
    c_y, c_x = img.shape[:2]
    c_y = c_y // 2
    c_x = c_x // 2
    shorter = min(img.shape[:2])
    if img.shape[0] <= img.shape[1]:
        img = img[c_y - shorter // 2: c_y + (shorter - shorter // 2) - 20, c_x - shorter // 2: c_x + (shorter - shorter // 2), :]
    else:
        img = img[c_y - shorter // 2: c_y + (shorter - shorter // 2), c_x - shorter // 2: c_x + (shorter - shorter // 2), :]

    return img


def map_and_dropnans(df, ans2idx):
    df['answer'] = df['answer'].map(ans2idx)
    df = df.dropna().reset_index(drop=True)
    df['answer'] = df['answer'].astype(int)
    return df

def map_answer_2_ids(train_df,val_df,test_df,args):

    #combine answers from all data splits and have the output neuron size equal to all possible answers
    if args.map_answers == 'combine':
        df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
        ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
        idx2ans = {idx:ans for ans,idx in ans2idx.items()}
        df['answer'] = df['answer'].map(ans2idx).astype(int)
        
        train_df = df[df['mode']=='train'].reset_index(drop=True)
        val_df = df[df['mode']=='val'].reset_index(drop=True)
        test_df = df[df['mode']=='test'].reset_index(drop=True)

    #output neuron size equal to 1000 top most frequent answer and have a neuron for the rest
    elif args.map_answers == 'top1000':
        answers=list(train_df['answer'].value_counts()[:1000].index)
        ans2idx = {ans:idx for idx,ans in enumerate(answers)}
        idx2ans = {idx:ans for ans,idx in ans2idx.items()}

        #top-1000 labels from training set from 0 to 999, drop the rest
        train_df = map_and_dropnans(train_df, ans2idx)
        val_df = map_and_dropnans(val_df, ans2idx)
        test_df = map_and_dropnans(test_df, ans2idx)
        
        # val_df['answer'] = val_df['answer'].map(ans2idx).fillna(1000).astype(int) 
        
    return train_df, val_df, test_df, ans2idx, idx2ans

class VQAMed(Dataset):
    def __init__(self, df, imgsize, tfm, args, mode): #mode = 'train'
        self.df = df
        self.tfm = tfm
        self.size = imgsize
        self.args = args
        if args.task == 'MLM':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif args.task == 'distillation':
            self.tokenizer = AutoTokenizer.from_pretrained(args.clinicalbert)
        self.mode = mode

        if self.mode == 'train':
            cats = self.df.category.unique()
            self.cats2ans = {c:i for i,c in enumerate(cats)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx,'img_id']
        question = self.df.loc[idx, 'question']

        answer = self.df.loc[idx, 'answer']

        # if self.mode == 'eval':
        #     tok_ques = self.tokenizer.tokenize(question)

        # if self.args.smoothing:
        #     answer = onehot(self.args.num_classes, answer)

        img = Image.open(path).convert('RGB')#cv2.imread(path)

        if self.tfm:
            img = self.tfm(img)

        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.args)

        if self.mode == 'train':
            cat = self.cats2ans[self.df.loc[idx, 'category']]
            return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long), path, torch.tensor(cat, dtype = torch.long)
        else:
            return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long), path




def calculate_bleu_score(preds,targets, idx2ans):
  bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split(), weights = [1]) for pred,target in zip(preds,targets)])
  return np.mean(bleu_per_answer)




def train_one_epoch(loader, model, optimizer, criterion, device, scaler, args, idx2ans):

    model.train()
    train_loss = []
    IMGIDS = []
    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave = False)
    for (img, question_token,segment_ids,attention_mask,target, imgid, category) in bar:

        img, question_token,segment_ids,attention_mask,target,category = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device), category.to(device)
        question_token = question_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                loss = loss_func(logits, target)
        else:
            logits, _, _ = model(img, question_token, segment_ids, attention_mask)
            if args.smoothing:
                loss = loss_func(logits, target, category)
            else:
                loss = loss_func(logits, target)
        if args.mixed_precision:
            scaler.scale(loss)
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        # if args.smoothing:
        #     TARGETS.append(target.argmax(1))
        # else:
        TARGETS.append(target)

        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)
        IMGIDS.append(imgid)

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        bar.set_description('train_loss: %.5f' % (loss_np))

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    IMGIDS = [i for sub in IMGIDS for i in sub]

    acc = (PREDS == TARGETS).mean() * 100.
    bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)

    return np.mean(train_loss), PREDS, acc, bleu, IMGIDS

#calc mock results for testing/debugging
def calc_MOCK_acc_and_bleu(val_df, PREDS, TARGETS,idx2ans,data_split):
    total_acc = 0.85 * 100.
        
    cats=val_df['category'].unique()
    accs = {}
    bleu = {}

    for c in cats:
        accs[c] = 0.80 * 100.
        bleu[c] = 0.70 * 100.

    
    final_accs = {}
    final_accs[f'{data_split}_total_acc'] = np.round(total_acc,4)

    final_bleus = {}
    total_bleu = 0.85
    final_bleus[f'{data_split}_total_bleu'] = np.round(total_bleu,4)

    for k,v in accs.items():
        final_accs[f'{data_split}_{k}_acc']=np.round(v,4)
        final_bleus[f'{data_split}_{k}_bleu']=np.round(v,4)
    return final_accs, final_bleus

def calc_acc_and_bleu(val_df, PREDS, TARGETS,idx2ans,data_split):

    total_acc = (PREDS == TARGETS).mean() * 100.
        
    cats=val_df['category'].unique()
    accs = {}
    bleu = {}

    for c in cats:
        accs[c] = (PREDS[val_df['category']==c] == TARGETS[val_df['category']==c]).mean() * 100.
        bleu[c] = calculate_bleu_score(PREDS[val_df['category']==c],TARGETS[val_df['category']==c],idx2ans)

    
    final_accs = {}
    final_accs[f'{data_split}_total_acc'] = np.round(total_acc,4)

    final_bleus = {}
    total_bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
    final_bleus[f'{data_split}_total_bleu'] = np.round(total_bleu,4)

    for k,v in accs.items():
        final_accs[f'{data_split}_{k}_acc']=np.round(v,4)
        final_bleus[f'{data_split}_{k}_bleu']=np.round(v,4)
    return final_accs, final_bleus

def validate(loader, model, criterion, device, scaler, args, val_df, idx2ans):

    model.eval()
    criterion.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, question_token,segment_ids,attention_mask,target, _) in bar:

            img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)


            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                logits, _ , _= model(img, question_token, segment_ids, attention_mask)
                if args.smoothing:
                    loss = criterion(logits, target,0)
                else:
                    loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            # if args.smoothing:
            #     TARGETS.append(target.argmax(1))
            # else:
            TARGETS.append(target)

            val_loss.append(loss_np)

            bar.set_description('val_loss: %.5f' % (loss_np))

        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    # Calculate total and category wise accuracy
    if args.category:
        acc = (PREDS == TARGETS).mean() * 100.
        bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
        return val_loss, PREDS, acc, bleu 
    else:
        final_accs, final_bleus  = calc_acc_and_bleu(val_df, PREDS, TARGETS,idx2ans,data_split='val')

    return val_loss, PREDS, final_accs, final_bleus 

def test(loader, model, criterion, device, scaler, args, val_df,idx2ans):

    model.eval()
    criterion.eval()

    PREDS = []
    TARGETS = []

    #test_loss = []
    #c= 0

    with torch.no_grad():
        for (img,question_token,segment_ids,attention_mask,target, _) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                    #loss = criterion(logits, target)
            else:
                logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                # if args.smoothing:
                #     loss = criterion(logits, target,0)
                # else:
                #     loss = criterion(logits, target)


            #loss_np = loss.detach().cpu().numpy()

            #test_loss.append(loss_np)

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            # if args.smoothing:
            #     TARGETS.append(target.argmax(1))
            # else:
            TARGETS.append(target)

            # if c ==3:
            #     break
            # c+=1
            

        #test_loss = np.mean(test_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    if args.category:
        acc = (PREDS == TARGETS).mean() * 100.
        bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
        return PREDS, acc, bleu
        #return test_loss, PREDS, acc, bleu
    else:
        final_accs, final_bleus  = calc_acc_and_bleu(val_df, PREDS, TARGETS,idx2ans,data_split='test')

    return PREDS, final_accs, final_bleus
    #return test_loss, PREDS, final_accs, final_bleus

def final_test(loader, all_models, device, args, val_df, idx2ans):

    PREDS = []

    with torch.no_grad():
        for (img,question_token,segment_ids,attention_mask,target) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

            for i, model in enumerate(all_models):
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                else:
                    logits, _, _ = model(img, question_token, segment_ids, attention_mask)

                if i == 0:
                    pred = logits.detach().cpu().numpy()/len(all_models)
                else:
                    pred += logits.detach().cpu().numpy()/len(all_models)

            PREDS.append(pred)

    PREDS = np.concatenate(PREDS)

    return PREDS

def test2020(loader, model, device, args):

    model.eval()

    PREDS = []

    with torch.no_grad():
        for (img, question_token, segment_ids, attention_mask) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                    # logits = model(img)
            else:
                logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                # logits = model(img)


            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)


    PREDS = torch.cat(PREDS).cpu().numpy()


    return PREDS


def val_img_only(loader, model, criterion, device, scaler, args, val_df, idx2ans):

    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, question_token,segment_ids,attention_mask,target, _) in bar:

            img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            # question_token = question_token.squeeze(1)
            # attention_mask = attention_mask.squeeze(1)


            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = model(img)
                    loss = criterion(logits, target)
            else:
                logits = model(img)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

            val_loss.append(loss_np)

            bar.set_description('val_loss: %.5f' % (loss_np))

        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    acc = (PREDS == TARGETS).mean() * 100.
    bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)



    return val_loss, PREDS, acc, bleu

def test_img_only(loader, model, criterion, device, scaler, args, test_df, idx2ans):

    model.eval()
    TARGETS = []
    PREDS = []
    test_loss = []

    with torch.no_grad():
        for (img, question_token, segment_ids, attention_mask, target, _) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            # question_token = question_token.squeeze(1)
            # attention_mask = attention_mask.squeeze(1)

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = model(img)
                    loss = criterion(logits, target)
            else:
                logits = model(img)
                loss = criterion(logits, target)


            pred = logits.softmax(1).argmax(1).detach()
            loss_np = loss.detach().cpu().numpy()

            PREDS.append(pred)
            TARGETS.append(target)
            test_loss.append(loss_np)

        test_loss = np.mean(test_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    acc = (PREDS == TARGETS).mean() * 100.
    bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)

    return test_loss, PREDS, acc, bleu



def train_img_only(loader, model, optimizer, criterion, device, scaler, args, idx2ans):

    model.train()
    train_loss = []
    PREDS = []
    TARGETS = []
    IMGIDS = []
    bar = tqdm(loader, leave = False)
    for (img, question_token,segment_ids,attention_mask,target, imgid) in bar:

        img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        # question_token = question_token.squeeze(1)
        # attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = loss_func(logits, target)
        else:
            logits = model(img)
            loss = loss_func(logits, target)

        if args.mixed_precision:
            scaler.scale(loss)
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        if args.smoothing:
            TARGETS.append(target.argmax(1))
        else:
            TARGETS.append(target)

        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)

        IMGIDS.append(imgid)
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        bar.set_description('train_loss: %.5f' % (loss_np))

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    IMGIDS = [i for sub in IMGIDS for i in sub]

    acc = (PREDS == TARGETS).mean() * 100.
    bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)

    return np.mean(train_loss), PREDS, acc, bleu, IMGIDS

class LabelSmoothByCategory(nn.Module):
    def __init__(self,  train_df, num_classes, device,smoothing = 0.1):
        super(LabelSmoothByCategory, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.train_df = train_df
        self.num_classes = num_classes
        self.device = device
        self.cross_entropy_criterion = nn.CrossEntropyLoss()

        self.computeCategoryTensors()


    def forward(self, x, target, category):
        if self.training:
            vecs = [self.idx2vector[xi.item()].clone().detach() for xi in category]
            v = []
            for i, xi in enumerate(vecs):
                idx = target[i]
                #print('idx', idx)
                xi[idx]  = self.confidence
                v.append(xi.unsqueeze(dim=0))
            soft_targets = torch.cat(v,0).to(self.device)
            #a = torch.cat([self.idx2vector[xi.item()].unsqueeze(dim=0) for xi in category],0)
            #a[target[:, 0], target[:, 1]] = self.confidence
            self.a=soft_targets
            return self.cross_entropy(x,soft_targets)
        else:
            #return torch.nn.functional.cross_entropy(x, target)
            # import IPython; IPython.embed(); exit(1)
            return self.cross_entropy_criterion(x, target)

    def computeCategoryTensors(self):
        #binary
        idx = self.train_df[self.train_df['category'] == 'binary']['answer'].unique()
        self.binary_tensor = torch.zeros(self.num_classes)
        self.binary_tensor[idx] = self.smoothing / len(idx)

        #plane
        idx = self.train_df[self.train_df['category'] == 'plane']['answer'].unique()
        self.plane_tensor = torch.zeros(self.num_classes)
        self.plane_tensor[idx] = self.smoothing / len(idx)

        #modality
        idx = self.train_df[self.train_df['category'] == 'modality']['answer'].unique()
        self.modality_tensor = torch.zeros(self.num_classes)
        self.modality_tensor[idx] = self.smoothing / len(idx)

        #organ
        idx = self.train_df[self.train_df['category'] == 'organ']['answer'].unique()
        self.organ_tensor = torch.zeros(self.num_classes)
        self.organ_tensor[idx] = self.smoothing / len(idx)

        #abnormality
        idx = self.train_df[self.train_df['category'] == 'abnormality']['answer'].unique()
        self.abnorm_tensor = torch.zeros(self.num_classes)
        self.abnorm_tensor[idx] = self.smoothing / len(idx)

        self.idx2vector = {0: self.plane_tensor, 1: self.modality_tensor, 2: self.binary_tensor,
        3: self.organ_tensor, 4: self.abnorm_tensor} #order by train_df.category.unique()


    def cross_entropy(self,pred, soft_targets):
        logsoftmax = nn.LogSoftmax(dim=1)
        a = - soft_targets * logsoftmax(pred)
        #print('a',a)
        return torch.mean(torch.sum(a, 1))

