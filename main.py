from data_preparation import DataPreparation
from regression import Regression


csv_path = 'vente_maillots_de_bain.csv'
data_prep = DataPreparation(csv_path)
regression_model = Regression(data_prep)
