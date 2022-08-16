import os
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import math
import random
import config
from tqdm import tqdm

from model import MTLModels
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

class StatisticalTools:
    def __init__(self):
        pass

    def ttest(self, value,n_sample):
        z = 1.96
        standard_error = math.sqrt((value*(1-value))/n_sample)
        return round(1.96*standard_error,4) #margin_of_error
    
    def add_me(self, data, df_results, rows):
        #COMMENT: "df.shape[0]" may chnage for the MTL models
        #COMMENT: I could get "df.shape[0]" from "data_dict[???head/data???]['rows']"
        df = pd.read_csv(config.DATA_PATH + '/' + str(config.INFO_DATA[data]['datasets']['train'].split('_')[0]) + '_merge' + '_processed.csv')
        me_col = [col for col in df_results.columns if 'me_' in col]
        for col in me_col:
            df_results[col] = df_results.loc[-rows:, [col[3:]]].apply(lambda x: self.ttest(x, df.shape[0]),axis=1)
            
        return df_results
        

#COMMENT: I should add a task column becase each model may retrieve three lines of results (one for each task/head)
class MetricTools:
    def __init__(self):
        pass
    
    def create_df_results(self):
        #COMMENT: I need to add the column heads as in in the table "Final Results" on google drive
        return pd.DataFrame(columns=['model', 
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
    def __init__(self):
        self.file_grid_preds = config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '_predictions' +'.csv'
        self.file_fold_preds = config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '_predictions_' + 'fold' +'.csv'
        if os.path.isfile(self.file_fold_preds):
            os.remove(self.file_fold_preds)
    
    
    def hold_epoch_preds(self, df_val, pred_val, targ_val, model_name, heads, drop_out, lr, batch_size, max_len, transformer, epoch, fold):
        # pred columns name
        #COMMENT: for MTL a need add head or group of heads
        self.pred_col = model_name + '_' + heads + '_' + str(drop_out) + '_' + str(lr) + '_' + str(batch_size) + '_' + str(max_len) + '_' + transformer + '_' + str(epoch)
        
        if epoch == 1:
            self.df = pd.DataFrame({'text':df_val[config.INFO_DATA[heads]['text_col']].values,
                                    'target':targ_val, 
                                    self.pred_col:pred_val})
        else:
            self.df[self.pred_col] = pred_val
        
    def concat_fold_preds(self):
        if os.path.exists(self.file_fold_preds):
            # join/merge predictions
            df_saved = pd.read_csv(self.file_fold_preds)
            self.df = pd.concat([df_saved, self.df], ignore_index=True)
        
        self.df.to_csv(self.file_fold_preds, index=False)
    
    def save_preds(self):
        df_fold_preds = pd.read_csv(self.file_fold_preds)
        #TODO: check if the file exist and print log
        
        if os.path.exists(self.file_grid_preds):
            df_grid_preds = pd.read_csv(self.file_grid_preds)
            # df_fold_preds = pd.concat([df_grid_preds, df_fold_preds], axis='text', ignore_index=True)
            df_fold_preds = pd.merge(df_grid_preds, df_fold_preds, on=['text','target'], how='outer')
            #TODO: what todo do if there is a old "gridsearch_predictions.csv" file? I believe I need some how delet it or add coninciodents coluns over the older vergion
            
        
        df_fold_preds.to_csv(self.file_grid_preds, index=False)
        #delet fold dataframe
        # if os.path.isfile(self.file_fold_preds):
        #     os.remove(self.file_fold_preds)
        
        #TODO: double check the funvtion
        #TODO: double check where a placed the function in side run()
        

#COMMENT: the CrossValidation need to reveive model_characteristics because super().save_predictions() needs it
class CrossValidation(MetricTools, StatisticalTools):
    def __init__(self, df_train, df_val, model_name, heads, max_len, transformer, batch_size, drop_out, lr, df_results, fold):
        super(CrossValidation, self).__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.model_name = model_name
        self.heads = heads
        self.max_len = max_len
        self.transformer = transformer
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.lr = lr
        self.df_results = df_results if isinstance(df_results, pd.DataFrame) else super().create_df_results()
        self.fold = fold
        
    def calculate_metrics(self, pred, targ, pos_label=1, average='binary'):
        return {
                'f1':metrics.f1_score(targ, pred, pos_label=pos_label, average=average), 
                'acc':metrics.accuracy_score(targ, pred), 
                'recall':metrics.recall_score(targ, pred, pos_label=pos_label, average=average), 
                'precision':metrics.precision_score(targ, pred, pos_label=pos_label, average=average)
                }
        
    def run(self):
        
        #COMMENT: heads cannot to be use to acess data because it will have multiple heads EXIST-DETOXIS-HatEval
        train_dataset = dataset.TransformerDataset(
            text=self.df_train[config.INFO_DATA[self.heads]['text_col']].values,
            target=self.df_train[config.INFO_DATA[self.heads]['label_col']].values,
            max_len=self.max_len,
            transformer=self.transformer
        )

        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size, 
            num_workers = config.TRAIN_WORKERS
        )

        val_dataset = dataset.TransformerDataset(
            text=self.df_val[config.INFO_DATA[self.heads]['text_col']].values,
            target=self.df_val[config.INFO_DATA[self.heads]['label_col']].values,
            max_len=self.max_len,
            transformer=self.transformer
        )

        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, 
            batch_size=self.batch_size, 
            num_workers=config.VAL_WORKERS
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MTLModels(self.transformer, self.drop_out, number_of_classes=self.df_train[config.INFO_DATA[self.heads]['label_col']].max()+1)
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

        #COMMENT: Do a need to change "num_train_steps" and "scheduler" for the MTL model?
        num_train_steps = int(len(self.df_train) / self.batch_size * config.EPOCHS)
        optimizer = AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )
        
        manage_preds = PredTools()
        
        for epoch in range(1, config.EPOCHS+1):
            pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
            train_metrics = self.calculate_metrics(pred_train, targ_train)
            
            #TODO: Save pred_val in the table
            pred_val, targ_val, loss_val = engine.eval_fn(val_data_loader, model, device)
            val_metrics = self.calculate_metrics(pred_val, targ_val)
            
            #COMMENT: self.heads must to be splitted in group_heads and head
            #COMMENT: the fuc must contain framework/model_name, group_heads and head - check logs
            #COMMENT: the fuc must contain decoder-base, encoder-base and input
            manage_preds.hold_epoch_preds(self.df_val, pred_val, targ_val, 
                                        self.model_name, self.heads,
                                        self.drop_out, self.lr, self.batch_size, self.max_len, self.transformer,
                                        epoch, self.fold
                                    )
            
            df_new_results = pd.DataFrame({'model':self.model_name,
                                            'data': self.heads, #COMMENT: add index for the dataset or dataset name directly [???]
                                            'epoch':epoch,
                                            'transformer':self.transformer,
                                            'max_len':self.max_len,
                                            'batch_size':self.batch_size,
                                            'lr':self.lr,
                                            'dropout':self.drop_out,
                                            'accuracy_train':train_metrics['acc'],
                                            'f1-score_train':train_metrics['f1'],
                                            'recall_train':train_metrics['recall'],
                                            'precision_train':train_metrics['precision'],
                                            'loss_train':loss_train,
                                            'accuracy_val':val_metrics['acc'],
                                            'me_accuracy_val':0,
                                            'f1-score_val':val_metrics['f1'],
                                            'me_f1-score_val':0,
                                            'recall_val':val_metrics['recall'],
                                            'me_recall_val':0,
                                            'precision_val':val_metrics['precision'],
                                            'me_precision_val':0,
                                            'loss_val':loss_val
                                        }, index=[0]
                            ) 
            
            self.df_results = pd.concat([self.df_results, df_new_results], ignore_index=True)
            
            tqdm.write("Epoch {}/{} f1-score_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f} f1-score_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(epoch, 
                                                                                                                                                                                        config.EPOCHS, 
                                                                                                                                                                                        train_metrics['f1'], train_metrics['acc'], loss_train, 
                                                                                                                                                                                        val_metrics['f1'], val_metrics['acc'], loss_val))
            #COMMENT: I may move everithing bellow out of the For loop and remove if and else
            # save predictions
            if config.EPOCHS == epoch:
                manage_preds.concat_fold_preds()
            
            # avg and save logs
            if self.fold == config.SPLITS and config.EPOCHS == epoch:
                self.df_results = super().avg_results(self.df_results)
                #COMMENT: the add_me inputs must be changed for MTL train "self.model_name" & "len(self.heads.split('-'))" for the last I can use"len(data_dict.keys()) "
                #COMMENT: preper and save table as on google drive - prepering code for MTL model
                self.df_results = super().add_me(self.heads, self.df_results, len(self.heads.split('-')))
                super().save_results(self.df_results)
                
                manage_preds.save_preds()

        return self.df_results
    
if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    skf = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)
    df_results = None

    #COMMENT: To think about tqdm code
    #COMMENT: add feature layers encoder, feature layers decoder
    
    inter_parameters = len(config.TRANSFORMERS) * len(config.MAX_LEN) * len(config.BATCH_SIZE) * len(config.DROPOUT) * len(config.LR) * config.SPLITS
    inter_models =  len(config.MODELS.keys()) * math.prod([len(items['decoder']['heads']) for items in config.MODELS.values()])
    grid_search_bar = tqdm(total=(inter_parameters*inter_models), desc='GRID SEARCH', position=0)

    # get model_name/framework_name such as 'STL', 'MTL0' and etc & parameters
    for model_name, model_characteristics in config.MODELS.items():
        
        # start model -> get datasets/heads
        #COMMENT: Here I can get the other models characteristics for the MTL models
        for group_heads in model_characteristics['decoder']['heads']:
            
            # Model script starts Here!
            for head in group_heads.split('-'):
                data_dict = dict()
                
                # load datasets & create StratifiedKFold splitter
                data_dict[head] = {}
                data_dict[head]['merge'] = pd.read_csv(config.DATA_PATH + '/' + str(config.INFO_DATA[head]['datasets']['train'].split('_')[0]) + '_merge' + '_processed.csv', nrows=config.N_ROWS)
                data_dict[head]['rows'] = data_dict[head]['merge'].shape[0]
                data_dict[head]['data_split'] = skf.split(data_dict[head]['merge'][config.INFO_DATA[head]['text_col']], data_dict[head]['merge'][config.INFO_DATA[head]['label_col']])
                
            
            # grid search
            for transformer in config.TRANSFORMERS:
                for max_len in config.MAX_LEN:
                    for batch_size in config.BATCH_SIZE:
                        for drop_out in config.DROPOUT:
                            for lr in config.LR:
                                
                                # split data
                                for fold, indexes in enumerate(zip(*[data_dict[d]['data_split'] for d in sorted(data_dict.keys())]), start=1):
                                    
                                    for data, index in zip(sorted(data_dict.keys()), indexes):
                                        data_dict[data]['train'] = data_dict[data]['merge'].loc[index[0]]
                                        data_dict[data]['val'] = data_dict[data]['merge'].loc[index[1]]
                                        
                                        #COMMENT: move code below out of last for loop
                                        #COMMENT: run must receice data_dict instead of data_dict[data]['train'] or data_dict[data]['val']
                                        tqdm.write(f'\nModel: {model_name} Heads: {group_heads} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Fold: {fold}/{config.SPLITS}')
                                        
                                        #COMMENT: I shouldn't pass "head" or "data" to run function
                                        cv = CrossValidation(data_dict[data]['train'],
                                                            data_dict[data]['val'],
                                                            model_name,
                                                            group_heads,  #COMMENT: I shouldn't pass "heads" to function
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
                    









        ## TOOLKIT
        # print('@'*100)
        # print(self.df_train.columns)
        
        # TASKS:tes
        #     1) CHECK THE TWO CLASSES [DONE]
        #     2) ADD REMAIN OF THE CODE [DONE]
        #     3) FIX tqdm [DONE]
        #     4) Check avg and save script part [DONE]]
        #     6) add confidence interval to the results [DONE]
        #     8) check TODOs in the script [DONE]
        #     7) check logs [DONE]
        #     5) TEST CODE [DONE]
        #     10) if I am saying the data as I meat to do [DONE]
        #     Y) push last code version to github [DONE]
        #     X) train models with initial code [doing]
        #     9) plan next steps [doing]
        
