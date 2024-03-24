import pandas as pd
import os
PREPARED_DATA_PATH = r"C:\Users\shami\Desktop\Master_Thesis_Data\Prepared_Datasets"
TRAIN_TEST_DATA_PATH = r"C:\Users\shami\Desktop\Master_Thesis_Data\Train_Test_Datasets"

DATA_DICT = {
    "Adult":"income_class",
    "cc_fraud_1":"Class",
    "cc_fraud_2":"Class",
    "cc_fraud_3":"Class",
    "cc_fraud_4":"Class",
    "cc_fraud_05":"Class",
    "Churn_Ecom":"target_class",
    "CoverType":"target",
    "GMC_Credit_Scoring":"SeriousDlqin2yrs",
    "Nursery":"class",
    "PTB_Online":"Revenue",
    "PTP_Data":"ordered",
    "Taiwan_Credit_Scoring":"Y",
    "Wine":"quality"
    
}

class Experiment_Dataset:
    def __init__(self
                 ,name: str
                 ,X_train: pd.DataFrame
                 ,y_train: pd.Series
                 ,X_test: pd.DataFrame
                 ,y_test: pd.Series
                 ,target_col_name: str):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.target_col_name = target_col_name

def fetch_dataset(d_name:str):
    #get filepath
    p_train = os.path.join(TRAIN_TEST_DATA_PATH,f"{d_name}_test.xlsx")
    p_test = os.path.join(TRAIN_TEST_DATA_PATH,f"{d_name}_train.xlsx")
    
    #get train test data
    df_train = pd.read_excel(p_train)
    df_test = pd.read_excel(p_test)
    
    target_col = DATA_DICT[d_name]
    
    # split them to X y
    X_train = df_train.drop(columns = target_col)
    y_train = df_train[target_col]

    X_test = df_train.drop(columns = target_col)
    y_test = df_train[target_col]
    
    return Experiment_Dataset(d_name, X_train, y_train, X_test, y_test, target_col)

def fetch_all_datasets():
    dfs = []
    for d in DATA_DICT:
        print(f"Processing {d}...")
        dfs.append(fetch_dataset(d))
        print(f"{d} fetched successfully")
    return dfs