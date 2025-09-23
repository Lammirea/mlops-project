import configparser
import os
import unittest
import sys
import pandas as pd

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(current_dir, "../..", "config.ini"))
print(f"Пытаемся загрузить конфиг из: {config_path}")
if os.path.exists(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
else:
    raise FileNotFoundError(f"Ошибка: файл {config_path} не найден")

from train import MultiModel

# Сделать так, чтобы для unit тестов загружались данные из папки unit_tests/data_for_tests
class TestTrainModels(unittest.TestCase):
    def setUp(self):
        """Инициализация перед каждым тестом."""
        self.model = MultiModel()

    def test_log_reg_model(self):
        """Тест обучения логистической регрессии."""
       
        res = self.model.log_reg(use_config=False, predict=False, max_iter=50)
        self.assertTrue(res)
        # Проверяем, что файл модели создан
        self.assertTrue(os.path.exists(self.model.log_reg_path))

    # def test_rand_forest(self):
    #     """Тест обучения модели RandomForest."""
    #     res = self.model.rand_forest(use_config=False, n_estimators=5, max_depth=3, predict=False)
    #     self.assertTrue(res)
    #     self.assertTrue(os.path.exists(self.model.rand_forest_path))

    def test_d_tree(self):
        """Тест обучения модели DecisionTree."""
        res = self.model.d_tree(use_config=False, max_depth=2, min_samples_split=5, predict=False)
        self.assertTrue(res)
        self.assertTrue(os.path.exists(self.model.d_tree_path))

    def test_gnb(self):
        """Тест обучения модели GaussianNB."""
        res = self.model.gnb(predict=False)
        self.assertTrue(res)
        self.assertTrue(os.path.exists(self.model.gnb_path))

    def test_predict_smoke(self):
        """Тест предсказания в режиме 'smoke' для логистической регрессии."""
        # Обучаем модель (если еще не обучена)
        self.model.log_reg(use_config=False, solver="lbfgs", max_iter=50, predict=False)
        result = self.model.predict("log_reg", "smoke")
        self.assertIn("test_score", result)
        self.assertGreaterEqual(result["test_score"], 0.0)
        self.assertLessEqual(result["test_score"], 1.0)

if __name__ == '__main__':
    unittest.main()
