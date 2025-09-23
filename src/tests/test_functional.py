import requests
import pytest
import time
import json
import os
from io import StringIO
import pandas as pd
import numpy as np
import configparser


config = configparser.ConfigParser()
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(current_dir, "../..", "config.ini"))
config.read(config_path, encoding="utf-8")

host = config["FASTAPI"]["host"]
port = config.getint("FASTAPI", "port")
# Базовый URL для тестирования (можно переопределить через переменные окружения)
BASE_URL = f'http://{host}:{port}'

class TestFunctionalAPI:
    
    #@pytest.fixture(autouse=True)
    
    def test_01_train_decision_tree_model(self):
        """Тест 1: Обучение модели Decision Tree"""
        response = requests.post(
            f"{BASE_URL}/train/",
            params={
                "model_type": "d_tree",
                "use_config": False,
                "max_depth": 5,
                "min_samples_split": 10,
                "predict_flag": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_trained"] == True
        assert data["model_type"] == "d_tree"
        print("✅ Decision Tree model trained successfully")
    
    def test_02_train_random_forest_model(self):
        """Тест 2: Обучение модели Random Forest"""
        response = requests.post(
            f"{BASE_URL}/train/",
            params={
                "model_type": "rand_forest",
                "use_config": False,
                "n_estimators": 10,
                "criterion": "gini",
                "predict_flag": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_trained"] == True
        assert data["model_type"] == "rand_forest"
        print("✅ Random Forest model trained successfully")
    
    def test_03_train_logistic_regression_model(self):
        """Тест 3: Обучение модели Logistic Regression"""
        response = requests.post(
            f"{BASE_URL}/train/",
            params={
                "model_type": "log_reg",
                "use_config": False,
                "solver": "liblinear",
                "max_iter": 50,
                "predict_flag": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_trained"] == True
        assert data["model_type"] == "log_reg"
        print("✅ Logistic Regression model trained successfully")
    
    def test_04_train_gaussian_nb_model(self):
        """Тест 4: Обучение модели Gaussian Naive Bayes"""
        response = requests.post(
            f"{BASE_URL}/train/",
            params={
                "model_type": "gnb",
                "use_config": False,
                "predict_flag": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_trained"] == True
        assert data["model_type"] == "gnb"
        print("✅ Gaussian Naive Bayes model trained successfully")
    
       
    def test_05_invalid_model_type(self):
        """Тест 5: Попытка обучить несуществующую модель"""
        response = requests.post(
            f"{BASE_URL}/train/",
            params={
                "model_type": "invalid_model",
                "use_config": False
            }
        )
        
        assert response.status_code == 400
        print("✅ Invalid model type handling works correctly")
       