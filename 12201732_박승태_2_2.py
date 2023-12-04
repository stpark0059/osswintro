import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def sort_dataset(dataset_df):
    sort = dataset_df.sort_values(by=['year'])
    return sort

def split_dataset(dataset_df):   
    dataset_df['salary'] *= 0.001
    train = dataset_df.loc[:1718]
    test = dataset_df.loc[1718:]
    
    X_train = train.drop('salary', axis=1)
    Y_train = train['salary']
    
    X_test = test.drop('salary', axis=1)
    Y_test = test['salary']
    
    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
    num_columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    return dataset_df[num_columns]

def train_predict_decision_tree(X_train, Y_train, X_test):
    train = DecisionTreeRegressor(random_state=20)
    train.fit(X_train, Y_train)
    
    data = train.predict(X_test)
    
    return data

def train_predict_random_forest(X_train, Y_train, X_test):
    train = RandomForestRegressor(random_state=20)
    train.fit(X_train, Y_train)
    
    data = train.predict(X_test)
    
    return data

def train_predict_svm(X_train, Y_train, X_test):
    train = make_pipeline(StandardScaler(), SVR())
    train.fit(X_train, Y_train)
    
    data = train.predict(X_test)
    
    return data

def calculate_RMSE(labels, predictions):
    ms = mean_squared_error(labels, predictions)
    
    rmse = np.sqrt(ms)
    
    return rmse

if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
    
    sorted_df = sort_dataset(data_df)    
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
    
    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)
    
    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))    
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))    
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
