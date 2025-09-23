import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import shutil
import sys
import time
import traceback
import yaml
import numpy as np
import warnings
import redis
from src.preprocess import DataMaker

warnings.filterwarnings("ignore")

from src.logger import Logger

SHOW_LOG = True

class Predictor:
    def __init__(self):
        # Инициализация логгера и конфигурации
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)

        # Попробуем читать config.ini в рабочей директории, иначе — в родительской
        config_candidates = [
            os.path.join(os.getcwd(), "config.ini"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.ini")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "config.ini"))
        ]
        loaded = False
        for cfg in config_candidates:
            if os.path.exists(cfg):
                self.config.read(cfg)
                self.log.info(f"Configuration loaded from: {cfg}")
                loaded = True
                break
        if not loaded:
            self.log.error("Configuration file config.ini not found in expected locations.")
            raise FileNotFoundError("config.ini not found")

        # Парсер аргументов
        self.parser = argparse.ArgumentParser(description="Predictor")
        self.parser.add_argument("-m", "--model", type=str, help="Select model", required=True,
                                 nargs="?", choices=["RAND_FOREST", "GNB", "LOG_REG", "D_TREE"],
                                 default="D_TREE")
        self.parser.add_argument("-t", "--tests", type=str, help="Select tests", required=True,
                                 nargs="?", choices=["smoke", "func", "db"],
                                 default="smoke")

        # Загрузка данных согласно конфигу (ожидаем секцию DATA с train_file/test_file)
        try:
            train_path = os.path.normpath(os.path.join(os.getcwd(), self.config["DATA"]["train_file"]))
            test_path = os.path.normpath(os.path.join(os.getcwd(), self.config["DATA"]["test_file"]))
        except KeyError as e:
            self.log.error(f"Missing DATA.train_file or DATA.test_file in config.ini: {e}")
            raise

        # Загружаем DataFrame'ы
        train_df = pd.read_csv(train_path, encoding='latin1', low_memory=False)
        test_df = pd.read_csv(test_path, encoding='latin1', low_memory=False)

        # Предобработка
        preprocess_data = DataMaker()
        X_train_raw, self.y_train = preprocess_data.preprocess_data(train_df)
        X_test_raw, self.y_test = preprocess_data.preprocess_data(test_df)
        self.feature_columns = list(X_train_raw.columns)

        # Pipeline (imputer + scaler)
        self.pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        self.X_train_scaled = self.pipeline.fit_transform(X_train_raw)
        self.X_test_scaled = self.pipeline.transform(X_test_raw)

        # Сохранение предобработчика (по желанию)
        self.project_path = os.path.join(os.getcwd(), "experiments")
        os.makedirs(self.project_path, exist_ok=True)
        self.preprocessor_path = os.path.join(self.project_path, "preprocessor.sav")
        try:
            with open(self.preprocessor_path, "wb") as f:
                pickle.dump({'pipeline': self.pipeline, 'feature_columns': self.feature_columns}, f)
        except Exception:
            self.log.error("Failed to save preprocessor: " + traceback.format_exc())

        self.log.info("Predictor is ready")

    def predict(self):
        args = self.parser.parse_args()
        # Загрузка модели из конфига по имени модели
        try:
            model_path = self.config[args.model]["path"]
        except KeyError:
            self.log.error(f"Model {args.model} not found in config.ini")
            sys.exit(1)

        try:
            with open(model_path, "rb") as f:
                classifier = pickle.load(f)
        except FileNotFoundError:
            self.log.error(f"Model file not found: {model_path}")
            sys.exit(1)
        except Exception:
            self.log.error("Failed to load model: " + traceback.format_exc())
            sys.exit(1)

        if args.tests == "smoke":
            try:
                score = classifier.score(self.X_test_scaled, self.y_test)
                self.log.info(f'{args.model} has {score} score')
            except Exception:
                self.log.error("Smoke test failed: " + traceback.format_exc())
                sys.exit(1)
            self.log.info(f'{model_path} passed smoke tests')

        elif args.tests == "func":
            tests_path = os.path.join(os.getcwd(), "src/unit_tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:
                    try:
                        data = json.load(f)
                        X_raw = pd.json_normalize(data, record_path=['X'])
                        y = pd.json_normalize(data, record_path=['y'])
                        X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
                        # Подстраиваемся по колонкам, если какие-то отсутствуют — fillna
                        X_for_pipeline = X_raw.reindex(columns=self.feature_columns, fill_value=np.nan)
                        X_scaled = self.pipeline.transform(X_for_pipeline)
                        score = classifier.score(X_scaled, y)
                        self.log.info(f'{args.model} has {score} score on {test}')
                    except Exception:
                        self.log.error("Func test failed: " + traceback.format_exc())
                        sys.exit(1)

                    # Сохранение результатов эксперимента (как в оригинале)
                    exp_data = {
                        "model": args.model,
                        "model_params": dict(self.config.items(args.model)),
                        "tests": args.tests,
                        "score": str(score),
                        "X_test_path": test,
                        "y_test_path": test,
                    }
                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    os.makedirs(exp_dir, exist_ok=True)
                    with open(os.path.join(exp_dir, "exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    try:
                        shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir, "exp_logfile.log"))
                    except Exception:
                        # лог-файл может отсутствовать — игнорируем
                        self.log.warning("Could not copy logfile.log to experiment dir")
                    try:
                        shutil.copy(model_path, os.path.join(exp_dir, f'exp_{args.model}.sav'))
                    except Exception:
                        self.log.warning("Could not copy model to experiment dir")

        elif args.tests == "db":
            # Вынесем работу с БД в отдельную функцию, но для простоты — вызываем inline
            try:
                predictions = classifier.predict(self.X_test_scaled)
            except Exception:
                self.log.error("Failed to generate predictions: " + traceback.format_exc())
                sys.exit(1)

            # Получаем параметры подключения из окружения, с безопасными дефолтами
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            try:
                redis_port = int(os.getenv('REDIS_PORT', 6379))
            except ValueError:
                redis_port = 6379
            # По умолчанию не ставим пароль — чтобы не приводить к ошибке аутентификации
            redis_password = os.getenv('REDIS_PASSWORD', None)
            try:
                redis_db = int(os.getenv('REDIS_DB', 0))
            except ValueError:
                redis_db = 0

            # Попытка подключиться к Redis — но не падаем при ошибке подключения.
            try:
                conn = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    db=redis_db,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    decode_responses=True,
                )
                # Проверим соединение (ping)
                conn.ping()
                redis_available = True
            except (redis.ConnectionError, redis.TimeoutError) as e:
                self.log.warning(f"Redis not available ({e}). Predictions will be saved locally instead of Redis.")
                redis_available = False
            except redis.RedisError as e:
                self.log.warning(f"Redis error ({e}). Predictions will be saved locally instead of Redis.")
                redis_available = False
            except Exception as e:
                self.log.warning(f"Unexpected error while connecting to Redis ({e}). Saving locally.")
                redis_available = False

            if redis_available:
                try:
                    # Удаляем старые предсказания
                    conn.delete('predictions')
                    for pred in predictions:
                        # безопасное преобразование: если pred не int, приведём к int где возможно
                        try:
                            val = int(pred)
                        except Exception:
                            val = str(pred)
                        conn.rpush('predictions', val)

                    predictions_list = conn.lrange('predictions', 0, -1)
                    self.log.info("PREDICTIONS WRITTEN TO REDIS")
                    for i, pred in enumerate(predictions_list):
                        # pred может быть bytes или str
                        if isinstance(pred, bytes):
                            decoded = pred.decode('utf-8', errors='ignore')
                        else:
                            decoded = str(pred)
                        self.log.info(f"Prediction {i + 1}: {decoded}")

                except Exception:
                    # На случай если при операций с redis случится ошибка — логируем и сохраняем локально
                    self.log.error("Error while operating on Redis: " + traceback.format_exc())
                    self._save_predictions_locally(predictions)
            else:
                # Сохранение локально как резервный вариант
                self._save_predictions_locally(predictions)

        return True

    def _save_predictions_locally(self, predictions):
        # Сохраняем предсказания в файл как fallback, чтобы не терять результат
        try:
            out_path = os.path.join(os.getcwd(), "predictions_fallback.json")
            serializable = []
            for pred in predictions:
                try:
                    serializable.append(int(pred))
                except Exception:
                    serializable.append(str(pred))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"predictions": serializable, "created_at": datetime.utcnow().isoformat()}, f, ensure_ascii=False, indent=2)
            self.log.info(f"Predictions saved locally to {out_path}")
        except Exception:
            self.log.error("Failed to save predictions locally: " + traceback.format_exc())


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()
