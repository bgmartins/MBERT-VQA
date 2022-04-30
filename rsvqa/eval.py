import argparse
from utils import seed_everything, VQAMed, train_one_epoch, validate, test, load_data, LabelSmoothing, map_answer_2_ids
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
#from torchtoolbox.transform import Cutout
import os
#import pytorch_lightning as pl
import warnings
from models.mmbert import Model

warnings.simplefilter("ignore", UserWarning)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Evaluate")

    parser.add_argument('--run_name', type = str, required = True, help = "run name for wandb")
    parser.add_argument('--data_dir', type = str, required = False, default = "../ImageClef-2019-VQA-Med", help = "path for data")
    parser.add_argument('--phili_dataset', action = 'store_true', default = False, help = "flag for phili dataset evaluation")
    parser.add_argument('--model_dir', type = str, required = False, default = "../ImageClef-2019-VQA-Med/mmbert/MLM/vqamed-roco-1_acc.pt", help = "path to load weights")
    parser.add_argument('--save_dir', type = str, required = False, default = "../ImageClef-2019-VQA-Med/mmbert", help = "path to save weights")
    parser.add_argument('--category', type = str, required = False, default = None,  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = False, help = "use pretrained weights or not")
    parser.add_argument('--mixed_precision', action = 'store_true', default = False, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")

    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 4, help = "number of workers")
    parser.add_argument('--epochs', type = int, required = False, default = 100, help = "num epochs to train")
    parser.add_argument('--train_pct', type = float, required = False, default = 1.0, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1.0, help = "fraction of validation samples to select")
    parser.add_argument('--test_pct', type = float, required = False, default = 1.0, help = "fraction of test samples to select")
    parser.add_argument('--map_answers', type=str, required = True,default='combine', choices=['combine', 'top1000'], help='how to map answers to indices for VQA as classification problem')

    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 28, help = "max length of sequence")
    parser.add_argument('--batch_size', type = int, required = False, default = 16, help = "batch size")
    parser.add_argument('--lr', type = float, required = False, default = 1e-4, help = "learning rate'")
    # parser.add_argument('--weight_decay', type = float, required = False, default = 1e-2, help = " weight decay for gradients")
    parser.add_argument('--factor', type = float, required = False, default = 0.1, help = "factor for rlp")
    parser.add_argument('--patience', type = int, required = False, default = 10, help = "patience for rlp")
    # parser.add_argument('--lr_min', type = float, required = False, default = 1e-6, help = "minimum lr for Cosine Annealing")
    parser.add_argument('--hidden_dropout_prob', type = float, required = False, default = 0.3, help = "hidden dropout probability")
    parser.add_argument('--smoothing', type = float, required = False, default = None, help = "label smoothing")

    parser.add_argument('--image_size', type = int, required = False, default = 224, help = "image size")
    parser.add_argument('--hidden_size', type = int, required = False, default = 312, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 30522, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default = 4, help = "num of layers")
    parser.add_argument('--num_vis', type = int, required = True, help = "num of visual embeddings")
    parser.add_argument('--task', type=str, default='MLM',
                        choices=['MLM', 'distillation'], help='task which the model was pre-trained on')
    parser.add_argument('--clinicalbert', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--dataset', type=str, default='VQA-Med', help='roco or vqamed2019')
    parser.add_argument('--cnn_encoder', type=str, default='resnet152', help='name of the cnn encoder')
    parser.add_argument('--use_relu', action = 'store_true', default = False, help = "use ReLu")
    parser.add_argument('--transformer_model', type=str, default='transformer',choices=['transformer', 'realformer', 'feedback-transformer'], help='name of the transformer model')
    parser.add_argument('--wandb', action = 'store_false', default = True, help = "record in wandb or not")

    args = parser.parse_args()
    
    model_name = args.model_dir.split('/')[-1]
    ds = 'phili' if args.phili_dataset else 'testset'
    print('Using wandb',args.wandb)
    if args.wandb:
        wandb.init(project='rs-vqa', name = f'eval-{model_name}-{ds}', config = args) #args.run_name

    seed_everything(args.seed)


    train_df, val_df, test_df = load_data(args)

    train_df, val_df, test_df, ans2idx, idx2ans = map_answer_2_ids(train_df,val_df,test_df,args)

    num_classes = len(ans2idx)

    args.num_classes = num_classes

    train_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    if args.phili_dataset:
        print('Testing on phili dataset')
        philidf = pd.read_csv(os.path.join(args.data_dir, 'testdf_phili.csv'))
        phili_ans_unique = list(philidf['answer'].unique())

        #remain the original mapping from answer to id and add the new answers from the phili_df
        not_common = [a for a in phili_ans_unique if a not in ans2idx]
        phili_ans2idx = {ans:num_classes + idx for idx,ans in enumerate(not_common)} #new answer ids start at the end of the original ones

        #rewrite over the original mappings and testdf
        ans2idx = {**ans2idx, **phili_ans2idx} #merge mappings
        idx2ans = {idx:ans for ans,idx in ans2idx.items()}
        philidf['answer'] = philidf['answer'].map(ans2idx).astype(int)

        #update path to imgs
        philidf['img_id'] = philidf['img_id'].apply(lambda x: os.path.join(args.data_dir, f'Images_HR', str(x) + '.jpg'))
        test_df = philidf
        

    else:
        print('Testing on orig test set')
    
    #import IPython; IPython.embed(); exit(0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(args)

    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)

    print('Loading model at ', args.model_dir)
    model.load_state_dict(torch.load(args.model_dir)) #,map_location=torch.device('cpu')

    model.to(device)

    if args.wandb:
        wandb.watch(model, log='all')


    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)


    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()


    test_tfm = transforms.Compose([transforms.Resize(224), #added with profs
                                   transforms.CenterCrop(224), #added with profstransforms.ToTensor(),
                                   transforms.ToTensor(), 
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    testdataset = VQAMed(test_df, imgsize = args.image_size, tfm = test_tfm, args = args, mode='test')

    testloader = DataLoader(testdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    best_acc1 = 0
    best_acc2 = 0
    best_loss = np.inf
    counter = 0

    predictions, acc, bleu = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)
    #test_loss, predictions, acc, bleu = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)

    res = {
            #'test_loss': test_loss,
            'learning_rate': optimizer.param_groups[0]["lr"],
            **bleu,
            **acc,               
        }
    if args.wandb:
        wandb.log(res)
    #test_df = test_df[:16]
    test_df['preds'] = predictions
    test_df['decode_preds'] = test_df['preds'].map(idx2ans)
    test_df['decode_ans'] = test_df['answer'].map(idx2ans)
    test_df.to_csv(f'{args.save_dir}/{model_name}_preds.csv', index = False)
    
    result = test_df[['img_id', 'decode_preds']]
    result['img_id'] = result['img_id'].apply(lambda x: x.split('/')[-1].split('.')[0])
    result.to_csv(f'{args.save_dir}/{model_name}_res.txt', index = False, header=False, sep='|')
    print('acc', acc)
    print('bleu', bleu)

            