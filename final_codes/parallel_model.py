#!/usr/bin/env python
# coding: utf-8



from pyexpat import model
from threading import local
import warnings
warnings.filterwarnings('ignore')

import re
import string
import itertools
from collections import Counter
import nltk
import pandas as pd
import numpy as np
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from torchsummaryX import summary
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

global max_report_len
max_report_len = 0

global local_path
local_path = None

global pre_e
pre_e = 0

'''Dataset class to be given as input to DataLoader'''
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.dataframe = dataframe
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return torch.tensor(row.X), row.CATOHE, row.Z

'''Collate function to pick the mask, category and inputs from the dataframe'''
def my_collate(data):
    global max_report_len

    Xs, Cats, Ys = zip(*data)    
    maxSize = max_report_len
    bz = len(Xs)
    
    y = torch.tensor(Ys, dtype=torch.float)
    cat = torch.tensor(Cats, dtype=torch.float)
    x = torch.zeros((bz, maxSize), dtype=torch.int)
    mask = torch.zeros((bz, maxSize), dtype=torch.bool)
    
    for i, X in enumerate(Xs):
        for j, wordIdx in enumerate(X):
            x[i][j] = wordIdx
            mask[i][j] = True
    
    return x, mask, cat, y

'''BoT model'''
class Model(nn.Module):    
    def __init__(self, num_words, out_classes, numCats, category_layers=-1, hidden_sizes=None ):
        super().__init__()
        self.category_layers = category_layers
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=300)
        prev_size = 300
        self.fc_layers = nn.ModuleList()
        if category_layers > -1:
            prev_size += numCats
        if category_layers > 0:
            for hsize in hidden_sizes:
                self.fc_layers.append(nn.Linear(prev_size, hsize))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(0.4))
                prev_size = hsize
        self.fc_final = nn.Linear(in_features=prev_size, out_features=out_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask, cat):
        try:
            x = self.embedding(x)
        except:
            print(torch.max(x))
            raise Exception('I know Python!')

        maskUnsq = mask.unsqueeze(-1)
        x = torch.sum(x*maskUnsq, dim=1)
        x = x/torch.sum(maskUnsq, dim=1)
        x[x != x] = 0
        if(self.category_layers != -1):
            x = torch.cat((cat, x), dim=-1)
        if(self.category_layers > 0):
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
        x = self.fc_final(x)
#         x = self.sigmoid(x)
        return x

'''Function to move the optimizer to GPU'''
def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

'''Function to train a model and save weights periodically'''
def train(model, train_loader, val_loader, test_loader, n_epochs, optimizer, dic, criterion, scheduler, class_sum):
    the_last_val_loss = 1000
    patience = 3
    trigger_times = 0
    min_delta = 0.0001
    global pre_e
    global local_path
    model = model.cuda()
    for epoch in range(n_epochs):
        iterations = 0
        model.train()
        train_loss = 0
        for x, mask, cat, y in train_loader: 
            x = x.cuda()
            mask = mask.cuda()           
            cat = cat.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            y_hat = model(x, mask, cat)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if iterations%500 == 0:             
                print('Iteration: {} \t Training Loss: {:.6f}'.format(iterations+1, train_loss/(iterations+1)))
                scheduler.step()
            iterations += 1
        train_loss = train_loss / len(train_loader)
        print('*--------*')
        print('Epoch: {} \t Training Loss: {:.6f}'.format(pre_e+1, train_loss))


        val_loss, p, r, f = eval_model(model, val_loader, criterion, class_sum)
        print('Epoch: {} \t Validation loss: {:.2f}, p: {:.2f}, r: {:.2f}, f1: {:.2f}'
              .format(pre_e+1, val_loss, p, r, f))

        model.train()
        
        dic['train_losses'].append(train_loss)
        dic['val_losses'].append(val_loss)
        dic['val_precisions'].append(p)
        dic['val_recalls'].append(r)
        dic['val_f1scores'].append(f)

        if((pre_e+1) % 6 == 0):
            print("Computing Test performance")
            test_loss, test_p, test_r, test_f = eval_model(model, test_loader, criterion, class_sum)
            print('Epoch: {} \t Test loss: {:.2f}, p: {:.2f}, r: {:.2f}, f1: {:.2f}'
              .format(pre_e+1, test_loss, test_p, test_r, test_f))
            model.train()
            dic['test_losses'].append(test_loss)
            dic['test_precisions'].append(test_p)
            dic['test_recalls'].append(test_r)
            dic['test_f1scores'].append(test_f)
            dic['e'] = pre_e+1
            dic['premodel'] = model.module.state_dict()
            dic['optimizer'] = optimizer.state_dict()
            dic['scheduler'] = scheduler.state_dict()
            model_state = pre_e+1

            torch.save(dic, local_path +'_bnorm_checkpoint_{}.pth'.format(pre_e+1))
            torch.save({'state': model_state}, local_path + '_bnorm_state.pth')
        
        # Early stopping
        if val_loss > the_last_val_loss or (the_last_val_loss-val_loss)<min_delta:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return model
        else:
            print('trigger times: 0')
            trigger_times = 0

        the_last_val_loss = val_loss
        pre_e += 1
    
    return model

'''Function to evaluate the performance of a model'''
def eval_model(model, loader, criterion, sample_weights=None):
    model = model.cuda()    
    model.eval()
    with torch.no_grad():
        y_pred = torch.LongTensor()
        y_true = torch.LongTensor()
        val_loss = 0
        for x, mask, cat, y in loader:
            x = x.cuda()
            mask = mask.cuda()           
            cat = cat.cuda()
            y = y.cuda()
            y_hat = model(x, mask, cat)
            loss = criterion(y_hat, y)
            y_hat = F.sigmoid(y_hat)
            y_hat = (y_hat > 0.5).int()
            y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
            y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
            val_loss += loss.item()
        val_loss = val_loss / len(loader)
        print(torch.sum(y_pred, dim=0).tolist())
        p, r, f, _ = precision_recall_fscore_support(np.transpose(y_true), np.transpose(y_pred), average='samples')
        print("Without weights: ", val_loss, p, r, f)
        p, r, f, _ = precision_recall_fscore_support(np.transpose(y_true), np.transpose(y_pred), average='samples', sample_weight=sample_weights)
        return val_loss, p, r, f

'''CNN-based models'''
class CNNModel(nn.Module):    
    def __init__(self, num_words, out_classes, nCats, category_layers=-1, hidden_sizes=None):
        super().__init__()

        global max_report_len

        self.category_layers = category_layers
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=300)
        self.conv = nn.Conv1d(300, 3, 250)
        self.maxpool = nn.MaxPool1d(3, 2)

        print('max_report_len=', max_report_len)

        prev_size = (3*(((max_report_len-252)//2)+1))
        print('prev_size=', max_report_len)

        self.fc_layers = nn.ModuleList()
        if category_layers > -1:
            prev_size += nCats
        if category_layers > 0:
            for hsize in hidden_sizes:
                self.fc_layers.append(nn.Linear(prev_size, hsize))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(0.4))
                prev_size = hsize
        self.fc_final = nn.Linear(in_features=prev_size, out_features=out_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask, cat):
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        maskUnsq = mask.unsqueeze(-1)
        x = x*maskUnsq    # 32 x max_size x 300
        # print(x.shape)
        x = torch.permute(x, (0, 2, 1)) # 32 x 300 x max_size
        # print(x.shape)
        x = F.relu(self.conv(x)) # 32 x 3 x (max_size-249)
        # print(x.shape)
        x = self.maxpool(x)   # 32 x 3 x ((max_size-252)//2)+1
        # print(x.shape)
        x = torch.flatten(x, 1, -1) # 32 x (3*(((max_size-252)//2)+1))
        # print(x.shape)
        if(self.category_layers != -1):
            x = torch.cat((cat, x), dim=1)
        if(self.category_layers > 0):
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
        x = self.fc_final(x)
        # x = self.sigmoid(x)
        return x

'''CNN 3-Conv1D model'''
class CNN3ConvModel(nn.Module):    
    def __init__(self, num_words, out_classes):
        super().__init__()

        global max_report_len

        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=300)
        self.conv1 = nn.Conv1d(300, 2, 250)
        self.bnorm1 = nn.BatchNorm1d(2)
        self.conv2 = nn.Conv1d(300, 3, 250)
        self.bnorm2 = nn.BatchNorm1d(3)
        self.conv3 = nn.Conv1d(300, 4, 250)
        self.bnorm3 = nn.BatchNorm1d(4)
        self.maxpool = nn.MaxPool1d(3, 2)
        #print((3*(((max_report_len-251)//2)+1)))
        prev_size = (3*(((max_report_len-251)//2)+1)) + (2*(((max_report_len-251)//2)+1)) + (4*(((max_report_len-251)//2)+1)) - 9
        self.fc_final = nn.Linear(in_features=prev_size, out_features=out_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask, cat):
        x = self.embedding(x)
        maskUnsq = mask.unsqueeze(-1)
        x = x*maskUnsq    # 32 x max_size x 300
        x = torch.permute(x, (0, 2, 1)) # 32 x 300 x max_size
        # print(x.shape)
        x1 = F.relu(self.bnorm1(self.conv1(x))) # 32 x 250 x (max_size-1)
        x1 = self.maxpool(x1)   # 32 x 250 x ((max_size-4)//2)+1
        # print(x1.shape)
        x2 = F.relu(self.bnorm2(self.conv2(x))) # 32 x 250 x (max_size-2)
        x2 = self.maxpool(x2)   # 32 x 250 x ((max_size-5)//2)+1
        # print(x2.shape)
        x3 = F.relu(self.bnorm3(self.conv3(x))) # 32 x 250 x (max_size-3)
        x3 = self.maxpool(x3)   # 32 x 250 x ((max_size-6)//2)+1
        # print(x3.shape)
        x1 = torch.flatten(x1, 1, -1) # 32 x (250*(((max_size-4)//2)+1))
        x2 = torch.flatten(x2, 1, -1) # 32 x (250*(((max_size-5)//2)+1))
        x3 = torch.flatten(x3, 1, -1) # 32 x (250*(((max_size-6)//2)+1))
        #print(x1.shape, x2.shape, x3.shape)
        x = torch.cat((x1, x2, x3), dim=-1)
        x = self.fc_final(x)
        # x = self.sigmoid(x)
        return x



def main(config):
    global local_path
    global max_report_len
    global pre_e
    
    '''Configure model parameters based on command-line args'''
    model_type = 0
    category_num = 0
    hsizes = None
    nEpochs = config["nEpochs"]
    if(config["model"]=="bot"):
        local_path = "model_cat_"
    elif(config["model"]=="cnn"):
        local_path = "cnn_model_cat_"
        model_type = 1
    elif(config["model"]=="cnn3"):
        local_path = "cnn3_model"
        model_type = 2
        
    if not config["category"]:
        local_path += "m1"
        category_num = -1
    else:
        if config["fullyConnected"] is None:
            category_num = 0
            local_path += "0"
        else:
            category_num = len(config["fullyConnected"])
            hsizes = config["fullyConnected"]
            local_path += str(category_num)
    local_path += "/"
    
    '''Load the preprocessed dataset and compute metadata'''
    dataset = pd.read_pickle('preprocessedMimic.pkl')

    dataset.drop('PREPROCESSED_TEXT', axis=1, inplace=True)

    print("Reading dataset done...")

    '''Function to roll-up ICD-9 codes'''
    def roll_up(x):
        return [code[:3] for code in literal_eval(str(x))]

    drop_max_len = 2200
    dataset = dataset[dataset['PAR_PREPROCESSED_TEXT'].map(len) < drop_max_len]

    dataset['ICD9_CODE'] = dataset.ICD9_CODE.apply(roll_up)

    uniqCodes = set(itertools.chain.from_iterable(dataset['ICD9_CODE']))
    code2idx = {code: idx for idx, code in enumerate(uniqCodes)}

    dataset['Y'] = dataset['ICD9_CODE'].apply(lambda codeList: [code2idx[code] for code in codeList])

    mlb = MultiLabelBinarizer()
    dataset['Z'] = list(mlb.fit_transform(dataset.Y))

    uniqueCats = set(dataset['CATEGORY'])

    ohe = OneHotEncoder(sparse=False)
    dataset['CATOHE'] = list(ohe.fit_transform(dataset['CATEGORY'].values.reshape(-1,1)))

    '''Split the data into train, val and test'''
    num_records = len(dataset)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    trainDf = dataset.iloc[:int(0.8*num_records)]
    valDf = dataset.iloc[int(0.8*num_records):int(0.9*num_records)]
    testDf = dataset.iloc[int(0.9*num_records):]

    print("Splitting dataset done...")

    '''Encode words using their vocabulary indices'''
    tokenFreq = Counter(itertools.chain.from_iterable(dataset['PAR_PREPROCESSED_TEXT']))

    tokenFreqTrain = Counter(itertools.chain.from_iterable(trainDf['PAR_PREPROCESSED_TEXT']))
    word2idxTrain = {k: i+1 for i, k in enumerate(tokenFreqTrain.keys())}
    word2idxTrain['unk'] = 0

    trainDf['X'] = trainDf['PAR_PREPROCESSED_TEXT'].apply(lambda words: [word2idxTrain[word] if word in word2idxTrain else 0 for word in words])
    valDf['X'] = valDf['PAR_PREPROCESSED_TEXT'].apply(lambda words: [word2idxTrain[word] if word in word2idxTrain else 0 for word in words])
    testDf['X'] = testDf['PAR_PREPROCESSED_TEXT'].apply(lambda words: [word2idxTrain[word] if word in word2idxTrain else 0 for word in words])

    dataset['REPORT_LENGTH'] = dataset.PAR_PREPROCESSED_TEXT.apply(lambda p: len(p))

    pre_e = 0

    max_report_len = np.max(dataset['REPORT_LENGTH'])

    '''Declare Dataloaders for each subset of data'''
    train_dataloader = torch.utils.data.DataLoader(MyDataset(trainDf), batch_size=32, shuffle=True, collate_fn = my_collate, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(MyDataset(valDf), batch_size=32, shuffle=True, collate_fn = my_collate, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(MyDataset(testDf), batch_size=32, shuffle=False, collate_fn = my_collate, num_workers=4)

    import math
    class_sum = trainDf['Z'].sum()
    second_min = np.amin(class_sum[class_sum != np.amin(class_sum)])
    class_sum[np.argmax(class_sum)] -= np.count_nonzero(class_sum==0)*second_min
    class_sum[class_sum == 0] = second_min

    vocabSize = len(word2idxTrain)
    numCodes = len(uniqCodes)
    criterion = nn.MultiLabelSoftMarginLoss(weight=torch.from_numpy(1/np.sqrt(class_sum)).cuda())
    # criterion = nn.BCELoss()

    '''Instantiate respective model'''
    model = None
    if(model_type == 0):
        model = Model(vocabSize+1, numCodes, len(uniqueCats), category_num, hsizes)
    elif(model_type == 1):
        model = CNNModel(vocabSize+1, numCodes, len(uniqueCats), category_num, hsizes)
    else:
        model = CNN3ConvModel(vocabSize+1, numCodes)
    model=torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= nEpochs*20 )

    '''Check if training needs to be resumed'''
    resume = False
    dic = {'train_losses':[], 'val_losses':[], 'val_precisions':[], 'val_recalls':[], 'val_f1scores':[], 
    'test_losses':[], 'test_precisions':[], 'test_recalls':[], 'test_f1scores':[], 'e':0, 'premodel':None, 'optimizer':None,
    'scheduler':None}
    if resume:
        model_state = torch.load(local_path +'_state.pth')['state']
        print('Loading checkpoint at model state {}'.format(model_state))
        dic = torch.load(local_path + '_checkpoint_{}.pth'.format(model_state))
        pre_e = dic['e']
        model.module.load_state_dict(dic['premodel'])
        optimizer.load_state_dict(dic['optimizer'])
        scheduler.load_state_dict(dic['scheduler'])
        optimizer_to(optimizer, "cuda:0")
        print('Resuming Training from epoch {}'.format(pre_e))
    else:
        model_state = 0
        pre_e =0
        print('Starting Training')

    # print(summary(model, torch.zeros((64,2198)).long(), torch.zeros((64,2198)).long(), torch.zeros((64,15)).long()))

    '''Train the model'''
    model = train(model, train_dataloader, val_dataloader, test_dataloader, nEpochs, optimizer, dic, criterion, scheduler, class_sum)

    '''Evaluate model performance on test set'''
    print(eval_model(model, test_dataloader, criterion, class_sum))
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Just an example",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", help="Model type")
    parser.add_argument("-c", "--category", action="store_true", help="Category included")
    parser.add_argument("-fc", "--fullyConnected", nargs="*", type=int)
    parser.add_argument("-n", "--nEpochs", nargs="?", default=30, type=int, help="Number of Epochs")
    args = parser.parse_args()
    config = vars(args)
    print(config)
    main(config)

