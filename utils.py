import os
import shutil
import pandas as pd
import gdown
from config import *

# TODO [X]  Adpt the MTL class to the new code 
# TODO [X] perform data preprocessing (class func)
# TODO [x] 1) remove not target language 
# TODO [X] 3) add task column
# TODO [X] 5)convert to csv; 
# TODO [X] 6) merge datasets 

class UtilsClass:
    '''Class with useful functions'''
    def __init__(self):
        self.run()
    
    def run(self):
        if DOWLOAD_DATA:
            print("--------Downloading data-----------")
            self.download_data()
            
        if PROCESS_DATA:
            print("--------Processing data-----------")
            self.prepare_data()
        
    
    def download_data(self):
        # create a data folder
        if os.path.exists(DATA_PATH):
            shutil.rmtree(DATA_PATH)
        os.makedirs(DATA_PATH)
        
        for folder_name, info in INFO_DATA.items():
            #download data folders to current directory
            gdown.download_folder(info['url'], quiet=True)
            sorce_folder = os.getcwd() + '/' + folder_name
            
            # move datasets to the data folder
            file_names = os.listdir(sorce_folder)
            for file_name in file_names:
                shutil.move(os.path.join(sorce_folder, file_name), DATA_PATH)
                
            # delete data folders from current directory
            shutil.rmtree(sorce_folder)

    def prepare_data(self, labels_col):
        files = [f for f in os.listdir(DATA_PATH) if 'processed' not in f]
        # merge_list = list()
        
        for file in files:
            task = [data_name for data_name in INFO_DATA.keys() if data_name.lower() in file.lower()]
            divide_columns = ',' if file[-4:] == '.csv' else '\t'
            df = pd.read_csv(os.path.join(DATA_PATH, '/' ,file), sep = divide_columns)
            
            #remove non-target language text
            if TARGET_LANGUAGE != INFO_DATA[task]['language']:
                if TARGET_LANGUAGE not in INFO_DATA[task]['language']:
                    print('Work language does not exist in the dataset: {}'.format(file))')
                    exit()
                else:
                    df = df.query('language == "{}"'.format(TARGET_LANGUAGE)).reset_index(drop=True)
                    
            # add task/had column to the dataframe
            df['head'] = task
            
            # convert label to number
            if type(INFO_DATA[task]['positive_class']) == str or type(INFO_DATA[task]['positive_class']) == '1':
                df[labels_col] = df[labels_col].apply(lambda x: 1 if x == INFO_DATA[task]['positive_class'] else 0)
            
            # save as a csv file
            df.to_csv(os.path.join(DATA_PATH, '/' ,file[:-4], '_processed.csv'), index=False)
                
            

    def merge(self):
        '''merge datasets'''
        
        for task in INFO_DATA.keys():
            files = [f for f in os.listdir(DATA_PATH) if 'processed' in f and task in f]
            if len(files) == 0:
                print('No processed files for task: {}'.format(task))
                exit()
            elif len(files) > 0:
                merge_file = [pd.read_csv(os.path.join(DATA_PATH, '/' , file)) for file in files]
                
                if len(files) == 1:
                    print('Only one processed file for task: {}'.format(task))
                
                else:
                    df = pd.concat(merge_file, ignore_index=True)
                    df.to_csv(os.path.join(DATA_PATH, '/' ,file[:-4], '_merge' ,'_processed.csv'), index=False)
                    print('Merged file for task: {}'.format(task))

























# def download_data(data_path, data_urls):
#     # create a data folder
#     if os.path.exists(data_path):
#         shutil.rmtree(data_path)
#     os.makedirs(data_path)
    
#     for folder_name, url in data_urls.items():
#         #download data folders to current directory
#         gdown.download_folder(url, quiet=True)
#         sorce_folder = os.getcwd() + '/' + folder_name
        
#         # move datasets to the data folder
#         file_names = os.listdir(sorce_folder)
#         for file_name in file_names:
#             shutil.move(os.path.join(sorce_folder, file_name), data_path)
            
#         # delete data folders from current directory
#         shutil.rmtree(sorce_folder)




# def process_OSACT2022_data(data_path, header, text_col, labels_col, index_col, columns_to_read):
#     arabic_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv2", keep_emojis = True)
#     files = [f for f in os.listdir(self.data_path) if 'processed' not in f]
    
#     for file in files:
#         if 'test' not in file:
#             df = pd.read_csv(self.data_path + '/' + file, sep='\t', header=None, usecols=columns_to_read)
#         else:
#             df = pd.read_csv(self.data_path + '/' + file, sep='\t', header=None, usecols=[0,1])
        
#         print(df)
        
#         if 'train' in file or 'dev' in file:
#             df[df.shape[1]+1] = df.iloc[:,-1].apply(lambda x: x if x == 'NOT_HS' else 'HS')
#             df = df[df.columns.tolist()[:-2] + df.columns.tolist()[-1:] + df.columns.tolist()[-2:-1]]
#             df.columns = header
#             df.replace(labels_col, inplace=True)
#             print(df.head())
#         else:
#             df.columns = header[:-3]
#             print('@'*20)
#             print(header[:-3])

#         text_col_processed = text_col + '_processed'
#         pass_value_config('DATASET_TEXT_PROCESSED', '\'' +  text_col_processed + '\'')
#         df[text_col_processed] = df.loc[:, text_col].apply(lambda x: arabic_prep.preprocess(x))
#         print(df.head())
        
#         dataset_name =  file[:-4] + '_processed' + '.txt'
#         variable = 'DATASET' + ['_TRAIN' if 'train' in file else '_DEV' if 'dev' in file else '_TEST'][0]
#         pass_value_config(variable, '\'' + dataset_name + '\'')
        
#         df.to_csv(self.data_path + '/' + dataset_name, index=False, sep='\t',  index_label=index_col)



# def pass_value_config(variable, value):
#     with open(config.CODE_PATH + '/' + 'config.py', 'r') as conf:
#         content = conf.read()
#         new = content.replace(variable + ' = ' + "''", variable + ' = ' +  value )
        
#     with open(config.CODE_PATH + '/' + 'config.py', 'w') as conf_new:
#         conf_new.write(new)


# def map_labels(df, labels_col):
#     for col, labels in labels_col.items():
#         df.replace({col:{number: string for string, number in labels.items()}}, inplace=True)
#     return df
