import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle 
from src.logger import logging
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    logging.info('Create an evaluation report of the model')
    try:
        report = {}

        for i in range(len(list(models))):
            logging.info('List of all the models')
            model = list(models.values())[i]

            logging.info('List of all the parameters')
            para=param[list(models.keys())[i]]
            logging.info(str(para))
            logging.info('Apply grid search for all the parameters')
            gs = GridSearchCV(model,para,cv=3)

            logging.info('using grid search to fit the models')
            gs.fit(X_train,y_train)

            logging.info('set the parameters with the best parameters obtained from grid search')
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    