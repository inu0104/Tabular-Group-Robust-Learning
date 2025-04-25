import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class GroupDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        # sample_id, group 유지 
        self.sample_ids = self.data["sample_id"].values
        self.groups = self.data["group"].values  

        # Features & Labels
        self.features = self.data.drop(columns=["target", "sample_id", "group"]).values 
        self.labels = self.data["target"].values 

        # PyTorch Tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.sample_ids = torch.tensor(self.sample_ids, dtype=torch.long)
        self.groups = torch.tensor(self.groups, dtype=torch.long)  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.groups[idx], self.sample_ids[idx], idx

    @staticmethod
    def from_dataframe(df):
        obj = GroupDataset.__new__(GroupDataset)
        obj.data = df.copy()

        obj.sample_ids = torch.tensor(df["sample_id"].values, dtype=torch.long)
        obj.groups = torch.tensor(df["group"].values, dtype=torch.long)
        obj.labels = torch.tensor(df["target"].values, dtype=torch.long)
        obj.features = torch.tensor(df.drop(columns=["target", "sample_id", "group"]).values, dtype=torch.float32)

        return obj


def get_column_names(names_file):
    columns = []
    with open(names_file, "r") as file:
        for line in file:
            if ":" in line:  
                col_name = line.split(":")[0].strip()
                columns.append(col_name)
    columns.append("income") 
    return columns

def group_rare_categories(df, col, top_n=5):
    freq = df[col].value_counts()
    categories_to_keep = freq.nlargest(top_n).index
    df[col] = df[col].apply(lambda x: x if x in categories_to_keep else "Other")
    return df

#########################################################################################################

def load_data(config):
    dataset = config["dataset"].lower()

    #ADULT
    if dataset == 'adult':
        
        train_file = "./dataset/adult/train_processed.csv"
        val_file = "./dataset/adult/valid_processed.csv"
        test_file = "./dataset/adult/test_processed.csv"
        
        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file)):
            
            names_path = "./dataset/adult/adult.names"
            train_path = "./dataset/adult/adult.data"
            test_path = "./dataset/adult/adult.test"

            columns = get_column_names(names_path)[15:]
            skip_rows = 1
            
            train = pd.read_csv(train_path, header=None, names=columns, na_values=" ?", skipinitialspace=True)
            test = pd.read_csv(test_path, header=None, names=columns, na_values=" ?", skipinitialspace=True, skiprows=skip_rows)
            
            train["target"] = train["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)
            test["target"] = test["income"].apply(lambda x: 1 if x.strip() == ">50K." else 0)
            train.drop(columns=["income"], inplace=True)
            test.drop(columns=["income"], inplace=True)
            
            categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "native-country"]
            group = ["sex", "race"]
            label_encoders = {}

            for col in categorical_columns + group: 
                label_enc = LabelEncoder()
                label_enc.fit(pd.concat([train[col], test[col]], axis=0).astype(str))  
                train[col] = label_enc.transform(train[col].astype(str))
                test[col] = label_enc.transform(test[col].astype(str))
                label_encoders[col] = label_enc  

            train["group"] = (train["sex"] * 2 + train["race"]).astype(int)
            test["group"] = (test["sex"] * 2 + test["race"]).astype(int)
            #train["group"] = train["sex"]
            #test["group"] = test["sex"]
            
            train.drop(columns=group, inplace=True)
            test.drop(columns=group, inplace=True)

            numerical_columns = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
            scaler = StandardScaler()
            scaler.fit(train[numerical_columns])
            train[numerical_columns] = scaler.transform(train[numerical_columns])
            test[numerical_columns] = scaler.transform(test[numerical_columns])
            
            full_train_df = train 
            test_df = test 

            train_df, val_df = train_test_split(
                full_train_df,
                test_size=0.4,
                stratify=full_train_df["target"],
                random_state=42
            )

            for df in [train_df, val_df, test_df]:
                df.reset_index(inplace=True)
                df.rename(columns={"index": "sample_id"}, inplace=True)

            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)

#########################################################################################################
    
    #COMPAS
    elif(dataset == 'compas'):
        
        data_path = "./dataset/compas/cox-violent-parsed_filt.csv"
        train_file = "./dataset/compas/train_processed.csv"
        val_file = "./dataset/compas/valid_processed.csv"
        test_file = "./dataset/compas/test_processed.csv"

        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file)):
            df = pd.read_csv(data_path)
         
            df["target"] = df["is_recid"].astype(int)
            df = df[df['target']!=-1]
  
            le = LabelEncoder()
            df["group"] = le.fit_transform(df["race"])
            # df["group1"] = le.fit_transform(df["sex"])
            # df["group2"] = le.fit_transform(df["race"])
            
            # df["group"] = (df["group1"] * 2 + df["group2"]).astype(int)
            
            drop_cols = [
                'id', 'name', 'first', 'last', 'sex', 'dob', 'is_recid', 'race', #'group1', 'group2',
                'c_jail_in', 'c_jail_out', 'c_days_from_compas', 'c_charge_desc',
                'r_offense_date', 'r_charge_desc', 'r_jail_in',
                'violent_recid', 'is_violent_recid',
                'vr_offense_date', 'vr_charge_desc',
                'decile_score.1', 'screening_date', 'event',
                'priors_count.1', 'start', 'end', 
                'decile_score','v_decile_score',
                'r_charge_degree' #data_leakeage
            ]
            df.drop(columns=drop_cols, inplace=True, errors='ignore')

            if 'type_of_assessment' in df.columns and df['type_of_assessment'].nunique() > 5:
                df = group_rare_categories(df, 'type_of_assessment', top_n=3)
            
            numerical_columns = [
                'age', 'juv_fel_count', 'juv_misd_count',  
                'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'r_days_from_arrest'
            ]
            df[numerical_columns] = df[numerical_columns].fillna(0)
            
            categorical_columns = [
                'age_cat', 'c_charge_degree', 'vr_charge_degree',
                'type_of_assessment', 'score_text', 'v_type_of_assessment', 'v_score_text'
            ]
            categorical_columns = [col for col in categorical_columns if col in df.columns]
            
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            df_final = df.reset_index(drop=True)
            
            feature_cols = [col for col in df_final.columns if col not in ["sample_id", "target", "group"]]
            df_final[feature_cols] = df_final[feature_cols].astype("float32")
            
            train_val_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42, stratify=df_final["target"])
            train_df, val_df = train_test_split(train_val_df, test_size=0.3, random_state=42, stratify=train_val_df["target"])

            for df in [train_df, val_df, test_df]:
                df.reset_index(inplace=True)
                df.rename(columns={"index": "sample_id"}, inplace=True)

            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)
   
#########################################################################################################  
            
    #GERMAN
    elif(dataset == 'german'):
        data_path = "./dataset/german/german_credit.csv"
        train_file = "./dataset/german/train_processed.csv"
        val_file = "./dataset/german/valid_processed.csv"
        test_file = "./dataset/german/test_processed.csv"
        
        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file)):
            df = pd.read_csv(data_path)
            
            df["target"] = df["class"].astype(int)
            df['group'] = df['Attribute13'].apply(lambda x: 1 if x>50 else 2)
            
            drop_cols = ["class"]
            df.drop(columns=drop_cols, inplace=True, errors="ignore")
            
            numerical_columns = ["Attribute" + str(i) for i in range(1, 21)]
            df[numerical_columns] = df[numerical_columns].fillna(0)
            
            categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
            if len(categorical_columns) > 0:
                df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            
            df_final = df.reset_index(drop=True)
            feature_cols = [col for col in df_final.columns if col not in ["sample_id", "target", "group"]]
            df_final[feature_cols] = df_final[feature_cols].astype("float32")
            
            train_val_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final["target"])
            train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df["target"])

            for df in [train_df, val_df, test_df]:
                df.reset_index(inplace=True)
                df.rename(columns={"index": "sample_id"}, inplace=True)

            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)
 
######################################################################################################### 
    
    #LAW
    elif(dataset == 'law'):
        data_path = "./dataset/law/law_school.csv"
        train_file = "./dataset/law/train_processed.csv"
        val_file = "./dataset/law/valid_processed.csv"
        test_file = "./dataset/law/test_processed.csv"

        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file)):
            df = pd.read_csv(data_path)
            df.dropna(axis=0, inplace=True)
            
            # (White:0, Non-White:1)
            race_dic = {'White': 0, 'Non-White': 1}
            df['race'] = df['race'].replace(race_dic)
            df["target"] = df["pass_bar"].astype(int)
            
            df['group'] = np.nan
            df.loc[(df['male'] == 0) & (df['race'] == 0), 'group'] = 0
            df.loc[(df['male'] == 0) & (df['race'] == 1), 'group'] = 1
            df.loc[(df['male'] == 1) & (df['race'] == 0), 'group'] = 2
            df.loc[(df['male'] == 1) & (df['race'] == 1), 'group'] = 3
            
            drop_cols = ["pass_bar", 'male', 'race']
            df.drop(columns=drop_cols, inplace=True, errors="ignore")
            
            exclude_cols = ["target", "group"]
            numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            numerical_columns = [col for col in numerical_columns if col not in exclude_cols]
            df[numerical_columns] = df[numerical_columns].fillna(0)
            
            categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
            categorical_columns = [col for col in categorical_columns if col not in exclude_cols]
            if len(categorical_columns) > 0:
                df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            
            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            
            df_final = df.reset_index(drop=True)
            feature_cols = [col for col in df_final.columns if col not in ["sample_id", "target", "group"]]
            df_final[feature_cols] = df_final[feature_cols].astype("float32")
            
            train_val_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42, stratify=df_final["target"])
            train_df, val_df = train_test_split(train_val_df, test_size=0.3, random_state=42, stratify=train_val_df["target"])

            for df in [train_df, val_df, test_df]:
                df.reset_index(inplace=True)
                df.rename(columns={"index": "sample_id"}, inplace=True)

            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)

#########################################################################################################  

    #CREDIT
    elif(dataset == 'credit'):
        data_path = "./dataset/credit/credit.csv"
        train_file = "./dataset/credit/train_processed.csv"
        val_file = "./dataset/credit/valid_processed.csv"
        test_file = "./dataset/credit/test_processed.csv"

        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file)):
            df = pd.read_csv(data_path, skiprows=1)
         
            df["target"] = df["default payment next month"].astype(int)
            df = df[df['target']!=-1]
  
            le = LabelEncoder()
            df["group"] = (df["AGE"] // 10).astype(int)
            
            drop_cols = [
                'ID','AGE','default payment next month'
            #     'id', 'name', 'first', 'last', 'sex', 'dob', 'is_recid', 'race',
            #     'c_jail_in', 'c_jail_out', 'c_days_from_compas', 'c_charge_desc',
            #     'r_offense_date', 'r_charge_desc', 'r_jail_in',
            #     'violent_recid', 'is_violent_recid',
            #     'vr_offense_date', 'vr_charge_desc',
            #     'decile_score.1', 'screening_date', 'event',
            #     'priors_count.1', 'start', 'end', 
            #     'decile_score','v_decile_score',
            #     'r_charge_degree' #data_leakeage
            ]
            df.drop(columns=drop_cols, inplace=True, errors='ignore')

            # if 'type_of_assessment' in df.columns and df['type_of_assessment'].nunique() > 5:
            #     df = group_rare_categories(df, 'type_of_assessment', top_n=3)
            
            numerical_columns = [
                'LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',  
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
            ]
            df[numerical_columns] = df[numerical_columns].fillna(0)
            
            categorical_columns = [
                'SEX', 'EDUCATION', 'MARRIAGE',
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
            ]
            categorical_columns = [col for col in categorical_columns if col in df.columns]
            
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            df_final = df.reset_index(drop=True)
            
            feature_cols = [col for col in df_final.columns if col not in ["sample_id", "target", "group"]]
            df_final[feature_cols] = df_final[feature_cols].astype("float32")
            
            train_val_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42, stratify=df_final["target"])
            train_df, val_df = train_test_split(train_val_df, test_size=0.3, random_state=42, stratify=train_val_df["target"])

            for df in [train_df, val_df, test_df]:
                df.reset_index(inplace=True)
                df.rename(columns={"index": "sample_id"}, inplace=True)

            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)

#########################################################################################################     
    train_df = pd.read_csv(train_file)
    
    train_dataset = GroupDataset(train_file)
    valid_dataset = GroupDataset(val_file)
    test_dataset = GroupDataset(test_file)

    batch_size = config.get("train_params", {}).get("batch_size", 32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, train_df
