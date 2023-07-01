import os
import sys
from logger import logging
from exception import CustomException
import pandas as pd
from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer

if __name__ == '__main__':
    
    # object intialization for data ingestion
    obj1=DataIngestion()
    train_data_path,test_data_path=obj1.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    # object  initialization for data transformation
    obj2=DataTransformation()
    train_data_arr,test_data_arr,_=obj2.initiate_data_transformation(train_data_path,test_data_path)

    # object initialization for model_training
    obj3=ModelTrainer()
    obj3.initiate_model_training(train_data_arr,test_data_arr)


