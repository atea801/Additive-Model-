from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


class Regression:
    def __init__(self, data_preparation_object):
        self.data_preparation_object = data_preparation_object
        self.model = LinearRegression()

        self.model.fit(data_preparation_object.x_train, data_preparation_object.y_train)

        y_train_predicted = self.model.predict(data_preparation_object.x_train)
        mean_train_absolute_error = np.mean(np.abs(y_train_predicted - data_preparation_object.y_train))
        print(f"Sur le jeu de train : {mean_train_absolute_error=:.2f}")

        y_test_predicted = self.model.predict(data_preparation_object.x_test)
        mean_test_absolute_error = np.mean(np.abs(y_test_predicted - data_preparation_object.y_test))
        print(f"Sur le jeu de test : {mean_test_absolute_error=:.2f}")

        self.show_model_predictions(y_train_predicted, y_test_predicted)

    def show_model_predictions(self, y_train_predicted, y_test_predicted):
        plt.figure(figsize=(15, 6))

        # TimeSeries Data
        plt.plot(self.data_preparation_object.dataset_df["Years"][:len(y_train_predicted)], self.data_preparation_object.y_train, "bo:", label="TimeSeries Data")

        # Fitted Additive Model
        plt.plot(self.data_preparation_object.dataset_df["Years"][:len(y_train_predicted)], y_train_predicted, "c", label="Fitted Additive Model")

        # True Future Data
        plt.plot(self.data_preparation_object.dataset_df["Years"][len(y_train_predicted):], self.data_preparation_object.y_test, "yo:", label="True Future Data", color='orange')

        # Forecasted Additive Model Data
        plt.plot(self.data_preparation_object.dataset_df["Years"][len(y_train_predicted):], y_test_predicted, "r", label="Forecasted Additive Model Data")

        # Calcul de l'intervalle de confiance
        conf_interval = 1.96 * np.std(y_test_predicted - self.data_preparation_object.y_test, ddof=1)
        upper_bound = y_test_predicted + conf_interval
        lower_bound = y_test_predicted - conf_interval

        # Plot de l'intervalle de confiance
        plt.fill_between(self.data_preparation_object.dataset_df["Years"][len(y_train_predicted):], lower_bound.squeeze(), upper_bound.squeeze(), color='gray', alpha=0.3, label="95% Confidence Interval")

        
        plt.legend()
        plt.show()
