# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from exception import CustomException
from logger import logging

from utils import save_object
from utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
        
        # imports libraries 
import os,sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from logger import logging
from exception import CustomException
from utils import save_object
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor

from utils import evaluate_model

# config the model training

@dataclass
class ModelTrainingConfig():
    trained_model_filepath=os.path.join("artifacts","model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_train_config=ModelTrainingConfig()
    def initiate_model_training(self,train_data_arr,test_data_arr):
        try:
            logging.info(" split the independent and dependent variables")
            X_train,y_train,X_test,y_test = (
                train_data_arr[:,:-1],
                train_data_arr[:,-1],
                test_data_arr[:,:-1],
                test_data_arr[:,-1]
            )
            # define models 
            models={
                "linearRegression":LinearRegression,
                "lasso":Lasso,
                "ridge":Ridge,
                "elasticnet":ElasticNet,
                "decisionTree":DecisionTreeRegressor
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            logging.info(f"model report: {model_report}")

            # choose  best model among all 
            best_model_score=max(sorted(model_report.values()))

            # find the best model name 
            best_model_name=list(model_report.keys())[
                list(model_report.values().index(best_model_score))
            ]
            best_model=models[best_model_name]

            print(f"best model name :{best_model} and r2_score :{best_model_score}")
            logging.info(f"best model name :{best_model} and r2_score :{best_model_score}")

            save_objects(
                file_path=self.ModelTrainingConfig.trained_model_filepath,
                obj=best_model
            )
        except Exception as e:
            logging.info("error occurred while training")
            raise CustomException(e,sys)

