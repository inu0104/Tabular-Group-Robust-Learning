import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class GroupDatasetForTabTransformer(Dataset):
    def __init__(self, df, categorical_cols, continuous_cols):
        self.df = df.reset_index(drop=True)
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols

        self.x_categ = df[categorical_cols].astype(int).values
        self.x_numer = df[continuous_cols].astype(np.float32).values
        self.y = df["target"].astype(int).values
        self.group = df["group"].astype(int).values
        self.sample_id = df["sample_id"].astype(int).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_categ = torch.tensor(self.x_categ[idx], dtype=torch.long)
        x_numer = torch.tensor(self.x_numer[idx], dtype=torch.float)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        g = torch.tensor(self.group[idx], dtype=torch.long)
        sid = torch.tensor(self.sample_id[idx], dtype=torch.long)
        return x_categ, x_numer, y, g, sid

def load_data_trans(config):
 
    dataset = config["dataset"]

    if dataset == 'adult':
        pass  # for now

    elif dataset == 'compas':
        data_path = "./dataset/compas/cox-violent-parsed_filt.csv"
        train_file = "./dataset/compas/train_transformer.csv"
        val_file = "./dataset/compas/valid_transformer.csv"
        test_file = "./dataset/compas/test_transformer.csv"

        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file)):
         
            df = pd.read_csv(data_path)
            df["target"] = df["is_recid"].astype(int)
            df = df[df['target'] != -1]

            le = LabelEncoder()
            df["group"] = le.fit_transform(df["race"])

            drop_cols = ['id', 'name', 'first', 'last', 'sex', 'dob', 'is_recid', 'race',
                         'c_jail_in', 'c_jail_out', 'c_days_from_compas', 'c_charge_desc',
                         'r_offense_date', 'r_charge_desc', 'r_jail_in', 'violent_recid',
                         'is_violent_recid', 'vr_offense_date', 'vr_charge_desc',
                         'decile_score.1', 'screening_date', 'event', 'priors_count.1',
                         'start', 'end', 'decile_score', 'v_decile_score', 'r_charge_degree']

            df.drop(columns=drop_cols, inplace=True, errors='ignore')

            if 'type_of_assessment' in df.columns and df['type_of_assessment'].nunique() > 5:
                df['type_of_assessment'] = df['type_of_assessment'].apply(lambda x: x if x in df['type_of_assessment'].value_counts().nlargest(3).index else 'other')

            numerical_columns = ['age', 'juv_fel_count', 'juv_misd_count',
                                 'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'r_days_from_arrest']

            df[numerical_columns] = df[numerical_columns].fillna(0)

            categorical_columns = ['age_cat', 'c_charge_degree', 'vr_charge_degree',
                                   'type_of_assessment', 'score_text', 'v_type_of_assessment', 'v_score_text']
            categorical_columns = [col for col in categorical_columns if col in df.columns]

            for col in categorical_columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

            df_final = df.reset_index(drop=True)
            feature_cols = [col for col in df_final.columns if col not in ["sample_id", "target", "group"]]
            df_final[feature_cols] = df_final[feature_cols].astype("float32")

            train_val_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42, stratify=df_final["target"])
            train_df, val_df = train_test_split(train_val_df, test_size=0.3, random_state=42, stratify=train_val_df["target"])

            for df_ in [train_df, val_df, test_df]:
                df_.reset_index(inplace=True)
                df_.rename(columns={"index": "sample_id"}, inplace=True)

            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)

        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        categorical_columns = ['age_cat', 'c_charge_degree', 'vr_charge_degree',
                            'type_of_assessment', 'score_text', 'v_type_of_assessment', 'v_score_text']
        categorical_columns = [col for col in categorical_columns if col in train_df.columns]
        continuous_columns = [col for col in train_df.columns if col not in categorical_columns + ["sample_id", "target", "group"]]

        # print("Categorical columns:", categorical_columns)
        # print("Cardinalities:", [train_df[col].nunique() for col in categorical_columns])
        # print("# Continuous features:", len(continuous_columns))

    train_dataset = GroupDatasetForTabTransformer(train_df, categorical_columns, continuous_columns)
    val_dataset = GroupDatasetForTabTransformer(val_df, categorical_columns, continuous_columns)
    test_dataset = GroupDatasetForTabTransformer(test_df, categorical_columns, continuous_columns)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, valid_loader, test_loader, train_df