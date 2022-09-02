import os
import re
import pandas as pd
import math
import config
from datetime import datetime

class StatisticalTools:
    def __init__(self):
        pass

    def ttest(self, value,n_sample):
        z = 1.96
        standard_error = math.sqrt((value*(1-value))/n_sample)
        return round(z*standard_error,4) #margin_of_error
    
    def add_margin_of_error(self, data_dict, heads, df_results):
        last_rows = -(len(heads)*config.EPOCHS)
        
        for col in df_results.columns:
            if 'me_' in col:
                df_results.loc[df_results.index[last_rows]:, col] = df_results.loc[df_results.index[last_rows]:, ['data', col[3:]]].apply(lambda x: self.ttest(x[col[3:]], data_dict[x['data']]['rows']),axis=1)
        
        return df_results

class MetricTools:
    def __init__(self, model_name, heads, transformer, max_len, batch_size, lr, drop_out, **kw):
        self.model_name = model_name
        self.heads = heads
        self.transformer = transformer
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
        self.drop_out = drop_out

    def new_lines_df(self, epoch, train_metrics, output_train, val_metrics, output_val):
        list_new_results = []
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
        return list_new_results

    def create_df_results(self):
        return pd.DataFrame(columns=['model',
                                        'heads',
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
    def __init__(self, data_dict, model_name, heads, drop_out, lr, batch_size, max_len, transformer):
        self.file_grid_preds = config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '_predictions' +'.csv'
        self.file_fold_preds = config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '_predictions' + '_fold' +'.csv'
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

def rename_logs():
    time_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    for file in os.listdir(config.LOGS_PATH):
        if not bool(re.search(r'\d', file)):
            os.rename(config.LOGS_PATH + '/' + file, config.LOGS_PATH + '/' + file[:-4] + '_' + time_str + file[-4:])
            
        