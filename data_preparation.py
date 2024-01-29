import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = 'vente_maillots_de_bain.csv'

class DataPreparation:
    def __init__(self, csv_path):
        self.dataset_df = pd.read_csv(csv_path)
        
        # 1. Transformer la colonne "Years" en format date
        self.dataset_df["Years"] = pd.to_datetime(self.dataset_df["Years"])

        # 2. Extraire le nom du mois sur une nouvelle colonne nommée "month_name"
        self.dataset_df["month_name"] = self.dataset_df["Years"].dt.month_name()

        # 3. Definir la limite des années 
        self.dataset_df = self.dataset_df[self.dataset_df["Years"].dt.year <= 2006]

        # 3. Transformer la nouvelle colonne "month_name" grâce au one-hot encoding
        self.month_dummies = pd.get_dummies(self.dataset_df["month_name"], prefix="month")
        for col in self.month_dummies.columns:
            self.dataset_df[col] = self.month_dummies[col]

        self.prepare_data()

    def prepare_data(self):
        number_of_rows = len(self.dataset_df)
               
        
        self.dataset_df["index_mesure"] = np.arange(0, number_of_rows, 1)

        dataset_train_df = self.dataset_df.iloc[: int(number_of_rows * 0.75)]
        dataset_test_df = self.dataset_df.iloc[int(number_of_rows * 0.75) :]

        self.x_train = dataset_train_df[["index_mesure"] + list(self.month_dummies.columns)].values
        self.y_train = dataset_train_df[["Sales"]].values

        self.x_test = dataset_test_df[["index_mesure"] + list(self.month_dummies.columns)].values
        self.y_test = dataset_test_df[["Sales"]].values

        

    def show_graph(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.dataset_df["Years"], self.dataset_df["Sales"], "x:")
        plt.show()
