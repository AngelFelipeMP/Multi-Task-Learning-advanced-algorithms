from config import *
from utils import download_data, process_OSACT2022_data

# args need to tell if it will download data AND/OR preprocess

if __name__ == "__main__":
    download_data(DATA_PATH,DATA_URL)
    
    
    process_OSACT2022_data(DATA_PATH, 
                    DATASET_COLUMNS, 
                    DATASET_TEXT, 
                    DATASET_CLASSES, 
                    DATASET_INDEX,
                    USEFUL_COLUMNS
    )

