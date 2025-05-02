import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np


def load_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e
    
def eval(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse, mae, r2


def main(alpha, l1_ratio):
    df = load_data()
    target = "quality"
    x = df.drop(columns=target)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42, test_size=0.2)

    mlflow.set_experiment("ElasticNet data modelling")
    with mlflow.start_run():
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio, random_state=42)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        rmse,mae,r2 = eval(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model,"trained_model")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--alpha","-a", type=float, default=0.2)
    args.add_argument("--l1_ratio","-l1", type=float, default=0.3)
    parsed_args = args.parse_args()
    # parsed_args.param1
    main(parsed_args.alpha, parsed_args.l1_ratio)
