import os
#Hiper-parameters
SPLITS = 5
EPOCHS = 5
MAX_LEN = [64]
DROPOUT = [0.1, 0.2, 0.3]
LR = [1e-5, 3e-5, 5e-5]
BATCH_SIZE = [32]
TRANSFORMERS = ['aubmindlab/araelectra-base-discriminator',
                'aubmindlab/bert-base-arabertv02-twitter',
                'xlm-roberta-base',
                'bert-base-multilingual-cased', 
                'aubmindlab/aragpt2-base',
                'asafaya/albert-large-arabic']

N_ROWS= None
SEED = 17
CODE_PATH = os.getcwd()
REPO_PATH = '/'.join(CODE_PATH.split('/')[0:-1])
DATA_PATH = REPO_PATH + '/' + 'data'

DATA_URL = ['']

LABELS = ['A', 'B', 'C']
DATASET_TEXT = 'text'
DATASET_TEXT_PROCESSED = 'text_processed'
DATASET_INDEX = 'index'
DATASET_COLUMNS = [DATASET_INDEX, DATASET_TEXT] + LABELS
DATASET_CLASSES = {DATASET_COLUMNS[2]:{'NOT_OFF':0, 'OFF': 1}, DATASET_COLUMNS[3]:{'NOT_HS':0, 'HS':1}, DATASET_COLUMNS[4]:{'NOT_HS':-1,'HS1':0,'HS2':1, 'HS3':2, 'HS4':3, 'HS5':4, 'HS6':5 }}
USEFUL_COLUMNS = [0,1,2,3]


DATASET_TRAIN = 'OSACT2022-sharedTask-train_processed.txt'
DATASET_DEV = 'OSACT2022-sharedTask-dev_processed.txt'
DATASET_TEST = 'OSACT2022-sharedTask-test-tweets_processed.txt'

DOMAIN_GRID_SEARCH = 'gridsearch'
DOMAIN_TRAIN = 'training'
DOMAIN_VALIDATION = 'validation'
DOMAIN_TRAIN_ALL_DATA = 'all_data_training'
DOMAIN_TEST = 'test'

TRAIN_WORKERS = 1
VAL_WORKERS = 1 
LOGS_PATH = REPO_PATH + '/' + 'logs'
