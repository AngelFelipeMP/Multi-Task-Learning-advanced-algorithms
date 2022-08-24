import os
#Hiper-parameters
SPLITS = 2
EPOCHS = 2
MAX_LEN = [64] #[128]
DROPOUT = [0.3]
LR = [5e-5]
BATCH_SIZE = [12]
TRANSFORMERS = ['dccuchile/bert-base-spanish-wwm-cased']

N_ROWS= 64
SEED = 17
CODE_PATH = os.getcwd()
REPO_PATH = '/'.join(CODE_PATH.split('/')[0:-1])
DATA_PATH = REPO_PATH + '/' + 'data'
LOGS_PATH = REPO_PATH + '/' + 'logs'

TARGET_LANGUAGE = 'es'
DOWLOAD_DATA = False
PROCESS_DATA = True


# LABELS = ['A', 'B', 'C']
# DATASET_TEXT = 'text'
# DATASET_TEXT_PROCESSED = 'text_processed'
# DATASET_INDEX = 'index'
# DATASET_COLUMNS = [DATASET_INDEX, DATASET_TEXT] + LABELS
# DATASET_CLASSES = {DATASET_COLUMNS[2]:{'NOT_OFF':0, 'OFF': 1}, DATASET_COLUMNS[3]:{'NOT_HS':0, 'HS':1}, DATASET_COLUMNS[4]:{'NOT_HS':-1,'HS1':0,'HS2':1, 'HS3':2, 'HS4':3, 'HS5':4, 'HS6':5 }}
# USEFUL_COLUMNS = [0,1,2,3]


# DATASET_TRAIN = 'OSACT2022-sharedTask-train_processed.txt'
# DATASET_DEV = 'OSACT2022-sharedTask-dev_processed.txt'
# DATASET_TEST = 'OSACT2022-sharedTask-test-tweets_processed.txt'

DOMAIN_GRID_SEARCH = 'gridsearch'
DOMAIN_TRAIN = 'training'
DOMAIN_VALIDATION = 'validation'
DOMAIN_TRAIN_ALL_DATA = 'all_data_training'
DOMAIN_TEST = 'test'

TRAIN_WORKERS = 1
VAL_WORKERS = 1 



INFO_DATA = {'DETOXIS': {
                    'task': 'toxicity detection',
                    'url':'https://drive.google.com/drive/folders/1KnFDh6oykkhW0h3AS1OhGqnjp-akH10J?usp=sharing',
                    'text_col':'comment',
                    'label_col':'toxicity',
                    'positive_class':1,
                    'metric':'F1-score',
                    'language':'es',
                    'datasets': {
                        'train': 'DETOXIS2021_train.csv',
                        'test': 'DETOXIS2021_test_with_labels.csv'
                    }
                },
            'EXIST': {
                    'task': 'sexism detection',
                    'url':'https://drive.google.com/drive/folders/1UlxE4jeze3tzfwrwrRywsP2nA4C96U13?usp=sharing',
                    'text_col':'text',
                    'label_col':'task1',
                    'positive_class':'sexist',
                    'metric':'Accuracy',
                    'language':'en-es',
                    'datasets': {
                        'train': 'EXIST2021_training.tsv',
                        'test': 'EXIST2021_test_with_labeled.tsv'
                    }
                },
            'HatEval': {
                    'task': 'hate speech detection',
                    'url':'https://drive.google.com/drive/folders/1XAcXmF-jerbQNy_nwzjkBuhjnU7TektN?usp=sharing',
                    'text_col':'text',
                    'label_col':'HS',
                    'positive_class':1,
                    'metric':'F1-score',
                    'language':'es',
                    'datasets': {
                        'train': 'HatEval2019_es_train.csv',
                        'dev':'HatEval2019_es_dev.csv',
                        'test': 'HatEval2019_es_test.csv'
                    }
                }
        }

MODELS = {
        'STL': {
            'decoder': {
                'base':'classifier',
                'heads':['EXIST', 'DETOXIS']},
            'encoder': {
                'base':'transformer', 
                'input':['text']}
            }
        }


# MODELS = {
#         'STL': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['EXIST', 'DETOXIS', 'HatEval']},
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']}
#             },
#         'MTL0': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']}
#             }
#         }


# MODELS = {
#         'STL': {
#             'heads': [1],
#             'data': ['EXIST', 'DETOXIS', 'HatEval'],
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']},
#             'decoder': 'classifier'
#             },
#         'MTL0': {
#             'heads': [2,3],
#             'data': ['EXIST', 'DETOXIS', 'HatEval'],
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']},
#             'decoder': 'classifier'
#             },
#         'MTL1': {
#             'heads': [2,3],
#             'data': ['EXIST', 'DETOXIS', 'HatEval'],
#             'encoder': {
#                 'model':'task-identification-encoder', 
#                 'input':['text', 'task-identification-text']},
#             'decoder': 'classifier'
#             },
#         'MTL2': {
#             'heads': [2,3],
#             'data': ['EXIST', 'DETOXIS', 'HatEval'],
#             'encoder': {
#                 'model':'task-identification-encoder', 
#                 'input':['text', 'task-identification-vector']},
#             'decoder': 'classifier'
#             },
#         'MTL3': {
#             'heads': [2,3],
#             'data': ['EXIST', 'DETOXIS', 'HatEval'],
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']},
#             'decoder': 'deep-classifier'
#             },
#         'MTL4': {
#             'heads': [2,3],
#             'data': ['EXIST', 'DETOXIS', 'HatEval'],
#             'encoder': {
#                 'model':'task-identification-encoder', 
#                 'input':['text', 'task-identification-text']},
#             'decoder': 'deep-classifier'
#             },
#         'MTL5': {
#             'heads': [2,3],
#             'data': ['EXIST', 'DETOXIS', 'HatEval'],
#             'encoder': {
#                 'model':'task-identification-encoder', 
#                 'input':['text', 'task-identification-vector']},
#             'decoder': 'deep-classifier'
#             }
#         }

