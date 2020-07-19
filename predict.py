import numpy as np
import csv
import sys
import pickle
from sklearn.neighbors import KNeighborsRegressor
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_re.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X

def import_train_data(train_X_path,train_Y_path):
    train_X = np.genfromtxt(train_X_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_path, delimiter=',', dtype=np.float64)
    return train_X,train_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def predict(test_X_file_path):

    # Load Test Data
    test_X = import_data(test_X_file_path)

    train_X,train_Y = import_train_data("train_X_re.csv","train_Y_re.csv")

    # Load Model Parameters
    neigh = KNeighborsRegressor()
    neigh.fit(train_X, train_Y)
    pred_Y = neigh.predict(test_X)
    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in regression.zip and imported properly.
    """

    write_to_csv_file(pred_Y, "predicted_test_Y_re.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_re.csv") 
