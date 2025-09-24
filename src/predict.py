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
import psycopg2
from psycopg2 import sql
import boto3
from botocore.client import Config
import io
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

        # Инициализируем MinIO клиент
        self.minio_client = self._initialize_minio_client()

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
            train_file_path = self.config["DATA"]["train_file"]
            test_file_path = self.config["DATA"]["test_file"]
        except KeyError as e:
            self.log.error(f"Missing DATA.train_file or DATA.test_file in config.ini: {e}")
            raise

        # Загружаем DataFrame'ы - сначала пробуем из MinIO, потом локально
        train_df = self._load_dataframe(train_file_path)
        test_df = self._load_dataframe(test_file_path)

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

    def _get_minio_config(self):
        """
        Получает конфигурацию MinIO из переменных окружения
        """
        endpoint_url = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
        access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        bucket_name = os.getenv('DVC_REMOTE_NAME', 'data')
        
        if endpoint_url and access_key and secret_key:
            return {
                'endpoint_url': endpoint_url,
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key,
                'bucket_name': bucket_name
            }
        return None

    def _initialize_minio_client(self):
        """
        Инициализирует клиент MinIO если конфигурация доступна
        """
        minio_config = self._get_minio_config()
        if minio_config:
            try:
                client = boto3.client(
                    's3',
                    endpoint_url=minio_config['endpoint_url'],
                    aws_access_key_id=minio_config['aws_access_key_id'],
                    aws_secret_access_key=minio_config['aws_secret_access_key'],
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1'
                )
                # Проверяем доступность бакета
                client.head_bucket(Bucket=minio_config['bucket_name'])
                self.log.info("MinIO клиент успешно инициализирован")
                return client
            except Exception as e:
                self.log.warning(f"MinIO недоступен: {e}")
                return None
        return None

    def _load_dataframe(self, file_path):
        """
        Загружает DataFrame из MinIO или локального файла
        """
        minio_config = self._get_minio_config()
        
        # Проверяем, является ли путь MinIO-путем или содержит ли он имя бакета
        if self.minio_client and minio_config and not file_path.startswith('http') and not os.path.isabs(file_path):
            # Считаем, что это имя файла в бакете
            try:
                self.log.info(f"Загружаем данные из MinIO: {file_path}")
                df = self._load_csv_from_minio(minio_config['bucket_name'], file_path)
                return df
            except Exception as e:
                self.log.warning(f"Ошибка загрузки из MinIO: {e}. Используем локальный файл.")
        
        # Загружаем из локального файла
        self.log.info(f"Загружаем данные из локального файла: {file_path}")
        local_path = os.path.normpath(os.path.join(os.getcwd(), file_path))
        return pd.read_csv(local_path, encoding='latin1', low_memory=False)

    def _load_csv_from_minio(self, bucket_name, file_key):
        """
        Загружает CSV файл из MinIO и возвращает pandas DataFrame
        """
        if not self.minio_client:
            raise Exception("MinIO клиент не инициализирован")
        
        try:
            # Загрузка файла
            response = self.minio_client.get_object(Bucket=bucket_name, Key=file_key)
            # Читаем данные из потока
            df = pd.read_csv(io.BytesIO(response['Body'].read()), encoding='latin1', low_memory=False)
            self.log.info(f"Данные успешно загружены из MinIO: {file_key}")
            return df
        except Exception as e:
            self.log.error(f"Ошибка загрузки CSV из MinIO: {e}")
            raise

    def predict(self):
        args = self.parser.parse_args()
        # Загрузка модели из конфига по имени модели
        try:
            model_path = self.config[args.model]["path"]
        except KeyError:
            self.log.error(f"Model {args.model} not found in config.ini")
            sys.exit(1)

        # Проверяем, является ли путь к модели MinIO-путем
        if model_path.startswith("minio://"):
            # Загружаем модель из MinIO
            try:
                model = self._load_model_from_minio(model_path)
                classifier = model
            except Exception as e:
                self.log.error(f"Failed to load model from MinIO: {e}")
                sys.exit(1)
        else:
            # Загружаем модель из локального файла
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
                        # Если модель была загружена из MinIO, копируем её в эксперимент
                        if model_path.startswith("minio://"):
                            # Загружаем модель из MinIO и сохраняем локально
                            model = self._load_model_from_minio(model_path)
                            with open(os.path.join(exp_dir, f'exp_{args.model}.sav'), 'wb') as f:
                                pickle.dump(model, f)
                        else:
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

            # Получаем параметры подключения из окружения
            postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
            try:
                postgres_port = int(os.getenv('POSTGRES_PORT', 5432))
            except ValueError:
                postgres_port = 5432
            postgres_db = os.getenv('POSTGRES_DB', 'postgres')
            postgres_user = os.getenv('POSTGRES_USER', 'postgres')
            postgres_password = os.getenv('POSTGRES_PASSWORD', None)

            # Попытка подключиться к PostgreSQL — но не падаем при ошибке подключения.
            try:
                conn = psycopg2.connect(
                    host=postgres_host,
                    port=postgres_port,
                    database=postgres_db,
                    user=postgres_user,
                    password=postgres_password,
                    connect_timeout=5
                )
                # Проверим соединение
                conn.cursor().execute("SELECT 1")
                postgres_available = True
                self.log.info(f"Connected to PostgreSQL at {postgres_host}:{postgres_port}")
            except (psycopg2.Error, Exception) as e:
                self.log.warning(f"PostgreSQL not available ({e}). Predictions will be saved locally instead of PostgreSQL.")
                postgres_available = False

            if postgres_available:
                try:
                    cursor = conn.cursor()
                    
                    # Создаем таблицу для предсказаний, если её нет
                    create_table_query = """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        prediction_value TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                    cursor.execute(create_table_query)
                    
                    # Удаляем старые предсказания
                    cursor.execute("DELETE FROM predictions")
                    
                    # Вставляем новые предсказания
                    for pred in predictions:
                        # безопасное преобразование
                        try:
                            val = str(int(pred))
                        except Exception:
                            val = str(pred)
                        
                        insert_query = "INSERT INTO predictions (prediction_value) VALUES (%s)"
                        cursor.execute(insert_query, (val,))
                    
                    conn.commit()
                    
                    # Читаем предсказания обратно для логирования
                    cursor.execute("SELECT id, prediction_value, created_at FROM predictions ORDER BY id")
                    predictions_rows = cursor.fetchall()
                    
                    self.log.info("PREDICTIONS WRITTEN TO POSTGRESQL")
                    for row in predictions_rows:
                        self.log.info(f"Prediction {row[0]}: {row[1]} (created: {row[2]})")
                    
                    cursor.close()
                    conn.close()

                except Exception:
                    # На случай если при операций с PostgreSQL случится ошибка — логируем и сохраняем локально
                    self.log.error("Error while operating on PostgreSQL: " + traceback.format_exc())
                    self._save_predictions_locally(predictions)
            else:
                # Сохранение локально как резервный вариант
                self._save_predictions_locally(predictions)

        return True

    def _load_model_from_minio(self, minio_path):
        """
        Загружает модель из MinIO
        """
        if not self.minio_client:
            raise Exception("MinIO клиент не инициализирован")
        
        try:
            # Извлекаем bucket_name и file_key из пути
            parts = minio_path.replace("minio://", "").split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Неверный формат MinIO пути: {minio_path}")
            
            bucket_name, file_key = parts
            
            # Загрузка файла
            response = self.minio_client.get_object(Bucket=bucket_name, Key=file_key)
            model_bytes = response['Body'].read()
            model = pickle.loads(model_bytes)
            self.log.info(f"Модель успешно загружена из MinIO: {file_key}")
            return model
        except Exception as e:
            self.log.error(f"Ошибка загрузки модели из MinIO: {e}")
            raise

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