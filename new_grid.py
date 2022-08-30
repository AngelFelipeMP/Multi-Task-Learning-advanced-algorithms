import os
import re
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import math
import random
import config
from tqdm import tqdm
from datetime import datetime

from torch.utils.data.dataset import ConcatDataset
from samplers import BatchSamplerTrain, BatchSamplerValidation
from model import MTLModels
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

#COMMENT:  I should make "data_dict[heads][data/head][...]" so I dont need to pass "heads"  I can use "data_dict[heads]"
#COMMENT: and when I want to acess "data/head" I shoud use "data_dict[heads][data/head]" or "data_dict.values()"
#COMMENT: ge the heads "list(data_dict[.keys())[0]"

class StatisticalTools:
    def __init__(self):
        pass

    def ttest(self, value,n_sample):
        z = 1.96
        standard_error = math.sqrt((value*(1-value))/n_sample)
        return round(z*standard_error,4) #margin_of_error
    
    # def add_me(self, data, df_results, rows):
    def add_margin_of_error(self, data_dict, heads, df_results):
        last_row = -(len(heads)*config.EPOCHS)
        
        for col in df_results.columns:
            if 'me_' in col:
                df_results.loc[df_results.index[last_row]:, col] = df_results.loc[df_results.index[last_row]:, ['data', col[3:]]].apply(lambda x: self.ttest(x[col[3:]], data_dict[x['data']]['rows']),axis=1)
        
        return df_results

class MetricTools:
    def __init__(self):
        pass
    
    def create_df_results(self):
        return pd.DataFrame(columns=['model',
                                        'heads'
                                        'data',
                                        'epoch',
                                        'transformer',
                                        'max_len',
                                        'batch_size',
                                        'lr',
                                        'dropout',
                                        'accuracy_train',
                                        'f1-score_train',
                                        'recall_train',
                                        'precision_train',
                                        'loss_train',
                                        'accuracy_val',
                                        'me_accuracy_val',
                                        'f1-score_val',
                                        'me_f1-score_val',
                                        'recall_val',
                                        'me_recall_val',
                                        'precision_val',
                                        'me_precision_val',
                                        'loss_val'
                                    ]
                    )
        
    def avg_results(self, df):
        return df.groupby(['model',
                            'heads',
                            'data',
                            'epoch',
                            'transformer',
                            'max_len',
                            'batch_size',
                            'lr',
                            'dropout'], as_index=False, sort=False)['accuracy_train',
                                                                    'f1-score_train',
                                                                    'recall_train',
                                                                    'precision_train',
                                                                    'loss_train',
                                                                    'accuracy_val',
                                                                    'me_accuracy_val',
                                                                    'f1-score_val',
                                                                    'me_f1-score_val',
                                                                    'recall_val',
                                                                    'me_recall_val',
                                                                    'precision_val',
                                                                    'me_precision_val',
                                                                    'loss_val'].mean()
    
    def save_results(self, df):
        df.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '.csv', index=False)
        
        
class PredTools:
    # def __init__(self, df_val, model_name, heads, drop_out, lr, batch_size, max_len, transformer):
    def __init__(self, data_dict, model_name, heads, drop_out, lr, batch_size, max_len, transformer):
        self.file_grid_preds = config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '_predictions' +'.csv'
        self.file_fold_preds = config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '_predictions' + '_fold' +'.csv'
        # self.df_val = df_val
        self.data_dict = data_dict
        self.heads = heads
        self.list_df = []
        self.model_name = model_name
        self.drop_out = drop_out
        self.lr = lr
        self.batch_size = batch_size
        self.max_len = max_len
        self.transformer = transformer
    
    def hold_epoch_preds(self, output_val, epoch):
        for index, head in enumerate(self.heads):
            # pred columns name
            pred_col = self.model_name + '_' + "-".join(self.heads) + '_' + head + '_' + str(self.drop_out) + '_' + str(self.lr) + '_' + str(self.batch_size) + '_' + str(self.max_len) + '_' + self.transformer + '_' + str(epoch)
            
            if epoch == 1:
                self.list_df.append(pd.DataFrame({'text':self.data_dict[head]['val'][config.INFO_DATA[head]['text_col']].values,
                                            'target':output_val[head]['targets'],
                                            pred_col:output_val[head]['predictions']}))
            else:
                self.list_df[index][pred_col] = output_val[head]['predictions']
        
        if epoch == config.EPOCHS:
            self.df_fold_preds = pd.concat(self.list_df, ignore_index=True)
        
    def concat_fold_preds(self):
        # concat folder's predictions
        if os.path.isfile(self.file_fold_preds):
            df_saved = pd.read_csv(self.file_fold_preds)
            self.df_fold_preds = pd.concat([df_saved, self.df_fold_preds], ignore_index=True)
            
        # save folder preds
        self.df_fold_preds.to_csv(self.file_fold_preds, index=False)
    
    def save_preds(self):
        if os.path.isfile(self.file_grid_preds):
            df_grid_preds = pd.read_csv(self.file_grid_preds)
            self.df_fold_preds = pd.merge(df_grid_preds, self.df_fold_preds, on=['text','target'], how='outer')
            
        # save grid preds
        self.df_fold_preds.to_csv(self.file_grid_preds, index=False)
        
        # delete folder preds
        if os.path.isfile(self.file_fold_preds):
            os.remove(self.file_fold_preds)

#COMMENT: I my move "rename_logs()" and "longer_dataset(" to CrossValidation 
#COMMENT: or I move the "calculate_metrics out of the Class"
#COMMENT: I must follow a code standard 
def rename_logs():
    time_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    for file in os.listdir(config.LOGS_PATH):
        if not bool(re.search(r'\d', file)):
            os.rename(config.LOGS_PATH + '/' + file, config.LOGS_PATH + '/' + file[:-4] + '_' + time_str + file[-4:])
            
            
def longer_dataset(data_dict):
    bigger = 0
    for head in data_dict.keys():
        if data_dict[head]['rows'] > bigger:
            bigger = data_dict[head]['rows']
            dataset = head
    
    return dataset
        

#COMMENT: the CrossValidation need to receive model_characteristics because super().save_preds() needs it
class CrossValidation(MetricTools, StatisticalTools):
    # def __init__(self, df_train, df_val, model_name, heads, max_len, transformer, batch_size, drop_out, lr, df_results, fold):
    def __init__(self, model_name, heads, data_dict, max_len, transformer, batch_size, drop_out, lr, df_results, fold):
        super(CrossValidation, self).__init__()
        # self.df_train = df_train
        # self.df_val = df_val
        
        self.model_name = model_name
        self.data_dict = data_dict
        self.heads = heads.split('-')
        
        self.max_len = max_len
        self.transformer = transformer
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.lr = lr
        self.df_results = df_results if isinstance(df_results, pd.DataFrame) else super().create_df_results()
        self.fold = fold
        
    # def calculate_metrics(self, pred, targ, pos_label=1, average='binary'):
    def calculate_metrics(self, output_train, pos_label=1, average='binary'):
        metrics_dict = {head:{} for head in self.heads}
        for head in self.heads:
            metrics_dict[head]['f1'] = metrics.f1_score(output_train[head]['targets'], output_train[head]['predictions'], pos_label=pos_label, average=average)
            metrics_dict[head]['acc'] = metrics.accuracy_score(output_train[head]['targets'], output_train[head]['predictions'])
            metrics_dict[head]['recall'] = metrics.recall_score(output_train[head]['targets'], output_train[head]['predictions'], pos_label=pos_label, average=average) 
            metrics_dict[head]['precision'] = metrics.precision_score(output_train[head]['targets'], output_train[head]['predictions'], pos_label=pos_label, average=average)
            
        return metrics_dict
        
    def run(self):
        self.concat = {'train_datasets':[], 'val_datasets':[]}
        # loading datasets
        for head in self.heads:
            self.concat['train_datasets'].append(dataset.TransformerDataset(
                text=self.data_dict[head]['train'][config.INFO_DATA[head]['text_col']].values,
                target=self.data_dict[head]['train'][config.INFO_DATA[head]['label_col']].values,
                max_len=self.max_len,
                transformer=self.transformer
                )
            )
            
            self.concat['val_datasets'].append(dataset.TransformerDataset(
                text=self.data_dict[head]['val'][config.INFO_DATA[head]['text_col']].values,
                target=self.data_dict[head]['val'][config.INFO_DATA[head]['label_col']].values,
                max_len=self.max_len,
                transformer=self.transformer
                )
            )

        # concat datasets
        concat_train = ConcatDataset(self.concat['train_datasets'])
        concat_val = ConcatDataset(self.concat['val_datasets'])
        
        # creating dataloaders
        train_data_loader = torch.utils.data.DataLoader(
            dataset=concat_train,
            sampler=BatchSamplerTrain(dataset=concat_train,batch_size=batch_size),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.TRAIN_WORKERS
        )

        val_data_loader = torch.utils.data.DataLoader(
            dataset=concat_val,
            batch_sampler=BatchSamplerValidation(dataset=concat_val,batch_size=batch_size),
            shuffle=False,
            num_workers=config.VAL_WORKERS
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = MTLModels(self.transformer, self.drop_out, number_of_classes=self.df_train[config.INFO_DATA[self.heads]['label_col']].max()+1)
        model = MTLModels(self.transformer, self.drop_out, self.heads, number_of_classes=2)
        model.to(device)
        
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # num_train_steps = int(len(self.df_train) / self.batch_size * config.EPOCHS)
        num_train_steps = int(len(self.data_dict[longer_dataset(self.data_dict)]['train']) / self.batch_size * config.EPOCHS)
        optimizer = AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )
        
        # create obt for save preds class
        manage_preds = PredTools(self.data_dict,
                                self.model_name, 
                                self.heads,
                                self.drop_out, 
                                self.lr, 
                                self.batch_size, 
                                self.max_len, 
                                self.transformer)
        
        for epoch in range(1, config.EPOCHS+1):
            # pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, self.heads)
            output_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, self.heads)
            # train_metrics = self.calculate_metrics(pred_train, targ_train)
            train_metrics = self.calculate_metrics(output_train)
            
            # pred_val, targ_val, loss_val = engine.eval_fn(val_data_loader, model, device, self.heads)
            output_val = engine.eval_fn(val_data_loader, model, device, self.heads)
            # val_metrics = self.calculate_metrics(pred_val, targ_val)
            val_metrics = self.calculate_metrics(output_val)
            
            # save epoch preds
            # manage_preds.hold_epoch_preds(pred_val, targ_val, epoch)
            manage_preds.hold_epoch_preds(output_val, epoch)
            
            #COMMENT: move the code below to class/function
            list_new_results =[]
            for head in self.heads:
                list_new_results.append(pd.DataFrame({'model':self.model_name,
                                                'heads':"-".join(self.heads),
                                                'data': head,
                                                'epoch':epoch,
                                                'transformer':self.transformer,
                                                'max_len':self.max_len,
                                                'batch_size':self.batch_size,
                                                'lr':self.lr,
                                                'dropout':self.drop_out,
                                                'accuracy_train':train_metrics[head]['acc'],
                                                'f1-score_train':train_metrics[head]['f1'],
                                                'recall_train':train_metrics[head]['recall'],
                                                'precision_train':train_metrics[head]['precision'],
                                                'loss_train':output_train[head]['loss'],
                                                'accuracy_val':val_metrics[head]['acc'],
                                                'me_accuracy_val':0,
                                                'f1-score_val':val_metrics[head]['f1'],
                                                'me_f1-score_val':0,
                                                'recall_val':val_metrics[head]['recall'],
                                                'me_recall_val':0,
                                                'precision_val':val_metrics[head]['precision'],
                                                'me_precision_val':0,
                                                'loss_val':output_val[head]['loss'],
                                            }, index=[0]
                                )
                )
            
            self.df_results = pd.concat([self.df_results, *list_new_results], ignore_index=True)
            
            tqdm.write("Epoch {}/{}".format(epoch,config.EPOCHS))
            for head in self.heads:
                tqdm.write("    Head: {:<8} f1-score_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f} f1-score_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(head,
                                                                                                                                                                                        train_metrics[head]['f1'], 
                                                                                                                                                                                        train_metrics[head]['acc'], 
                                                                                                                                                                                        output_train[head]['loss'], 
                                                                                                                                                                                        val_metrics[head]['f1'], 
                                                                                                                                                                                        val_metrics[head]['acc'], 
                                                                                                                                                                                        output_val[head]['loss']))
        
        # save a fold preds
        manage_preds.concat_fold_preds()
            
        # avg and save logs
        if self.fold == config.SPLITS:
            self.df_results = super().avg_results(self.df_results)
            self.df_results = super().add_margin_of_error(self.data_dict, self.heads, self.df_results)
            super().save_results(self.df_results)
            
            # save all folds preds "gridsearch"
            manage_preds.save_preds()

        return self.df_results
    
if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    #rename old log files adding date YMD-HMS
    rename_logs()

    #COMMENT: add feature layers encoder, feature layers decoder @
    inter_parameters = len(config.TRANSFORMERS) * len(config.MAX_LEN) * len(config.BATCH_SIZE) * len(config.DROPOUT) * len(config.LR) * config.SPLITS
    inter_models =  len(config.MODELS.keys()) * math.prod([len(items['decoder']['heads']) for items in config.MODELS.values()])
    grid_search_bar = tqdm(total=(inter_parameters*inter_models), desc='GRID SEARCH', position=0)
    
    # skf = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)
    # skf = RepeatedStratifiedKFold(n_splits=config.SPLITS, n_repeats=int(inter_parameters/config.SPLITS), random_state=config.SEED)
    df_results = None

    # get model_name/framework_name such as 'STL', 'MTL0' and etc & parameters
    #COMMENT: I should send everything "model_name, model_characteristics" together to "CrossValidation()"
    #COMMENT: related comment below
    for model_name, model_characteristics in config.MODELS.items():
        
        # start model -> get datasets/heads
        #COMMENT: Here I can get the other models characteristics for the MTL models @
        for group_heads in model_characteristics['decoder']['heads']:
            
            # Model script starts Here!
            data_dict = dict()
            for head in group_heads.split('-'):
                
                # load datasets & create StratifiedKFold splitter
                data_dict[head] = {}
                data_dict[head]['merge'] = pd.read_csv(config.DATA_PATH + '/' + str(config.INFO_DATA[head]['datasets']['train'].split('_')[0]) + '_merge' + '_processed.csv', nrows=config.N_ROWS)
                data_dict[head]['rows'] = data_dict[head]['merge'].shape[0]
                data_dict[head]['skf'] = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)
                # data_dict[head]['data_split'] = skf.split(data_dict[head]['merge'][config.INFO_DATA[head]['text_col']], data_dict[head]['merge'][config.INFO_DATA[head]['label_col']])
            
            # grid search
            for transformer in config.TRANSFORMERS:
                for max_len in config.MAX_LEN:
                    for batch_size in config.BATCH_SIZE:
                        for drop_out in config.DROPOUT:
                            for lr in config.LR:
                                
                                # split data
                                # for fold, indexes in enumerate(zip(*[data_dict[d]['data_split'] for d in sorted(data_dict.keys())]), start=1):
                                for fold, indexes in enumerate(zip(*[data_dict[d]['skf'].split(data_dict[d]['merge'][config.INFO_DATA[d]['text_col']], data_dict[d]['merge'][config.INFO_DATA[d]['label_col']]) for d in sorted(data_dict.keys())]), start=1):
                                    
                                    for data, index in zip(sorted(data_dict.keys()), indexes):
                                        data_dict[data]['train'] = data_dict[data]['merge'].loc[index[0]]
                                        data_dict[data]['val'] = data_dict[data]['merge'].loc[index[1]]
                                        
                                    tqdm.write(f'\nModel: {model_name} Heads: {group_heads} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Fold: {fold}/{config.SPLITS}')
                                    
                                    cv = CrossValidation(model_name, 
                                                        group_heads, #COMMENT: I shouldn't pass "heads" to function I could get it from data_dict if I add it as first key to data_dict["EXIST-DETOXIS-HatEval"] @
                                                        data_dict, 
                                                        max_len, 
                                                        transformer, 
                                                        batch_size, 
                                                        drop_out,
                                                        lr,
                                                        df_results,
                                                        fold
                                    )
                                    
                                    df_results = cv.run()
                                    grid_search_bar.update(1)
                    
                                    if fold == config.SPLITS:
                                        break









        ## TOOLKIT
        # print('@'*100)
        # print(self.df_train.columns)
        
        # TASKS:tes
        #     1) check COMMENTS [DONE]
        #     2) check paper and drive notes [DONE]
        #     3) check test [DONE]
        #     4) reading DataLoader MTL web article [DONE]
        #     5) plan the MTL implementation [DONE]
        #     6) Start implementation [DONE]
        #     7) Write dataloader script [DONE]
        #     8) add heads in the dataloader output[X]
        #     9) check model.py, engine.py and dataloader.py [DONE]
        #     10) Adapt new_grid_search for MTL - Dataset/DataLoader [DONE]
        #     11) Adapt new_grid_search for MTL - remaining [DONE]
        #     12) Check adaptation of new_grid.py [DONE]
        
        #     13) Run script and fix problems new_grid.py []
                    # - Run script and fix errors [X]
                    # - check logs/tables --> Bug skf.split --> check logs --> [X]
                    
                    # - print import output - add resuts, avg and so on
                    # - printe model structure for
                    # - check backpropagation
                    
        #     14) Break the script into utils.py and grid_search.py []
        #     15) Move part of the run code to a new class or func[]
        #     16) Double check utils.py and grid_search.py []
        #     17) Run utils.py and grid_search.py []
        #     18) Check the results from the experiment that I let running []
        #     19) Run middle lgth test with the code adapted to MTL []
