import configparser
import os
import pandas as pd
import numpy as np
import sys
import traceback
import boto3
from botocore.client import Config
import io

from logger import Logger

SHOW_LOG = True

class DataMaker:
    def __init__(self, to_show=True) -> None:
        # Logger: second parameter is "enable" in Logger
        logger = Logger(SHOW_LOG, to_show)
        if to_show:
            logger.clear_log_file()
        self.log = logger.get_logger(__name__)

        # Инициализируем парсер конфигурации прежде чем читать
        self.config = configparser.ConfigParser()

        # Получаем директорию текущего файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Формируем путь на уровень выше (если config.ini в родительской папке)
        self.config_path = os.path.abspath(os.path.join(current_dir, "..", "config.ini"))

        # Логируем путь к конфигу
        self.log.debug(f"Пытаемся загрузить конфиг из: {self.config_path}")

        if os.path.exists(self.config_path):
            # Читаем конфигурацию
            try:
                with open(self.config_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    # configparser может читать из file-like объекта
                    self.config.read_file(f)
                self.log.info("Конфигурация успешно загружена (utf-8-sig)")
            except Exception as e:
                self.log.error(f"Ошибка чтения config.ini: {e}")
                raise

        else:
            error_msg = f"Ошибка: файл {self.config_path} не найден"
            self.log.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Проверяем доступность MinIO и инициализируем клиент
        self.minio_client = self._initialize_minio_client()
        
        # Папка проекта для данных (по умолчанию в рабочей директории)
        self.project_path = os.path.join(os.getcwd(), "data")
        # Создадим папку, если её нет
        try:
            os.makedirs(self.project_path, exist_ok=True)
        except Exception:
            # Если не удалось создать папку — логируем и продолжим (файловые операции потом могут упасть)
            self.log.warning(f"Не удалось создать папку для данных: {self.project_path}")

        # Пути для сохранения предобработанных данных
        self.train_path = [
            os.path.join(self.project_path, "preprocessed_train_X.csv"),
            os.path.join(self.project_path, "preprocessed_train_y.csv")
        ]
        self.test_path = [
            os.path.join(self.project_path, "preprocessed_test_X.csv"),
            os.path.join(self.project_path, "preprocessed_test_y.csv")
        ]
        self.log.info("DataMaker is ready")

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

    def _save_csv_to_minio(self, df, bucket_name, file_key):
        """
        Сохраняет DataFrame в CSV и загружает в MinIO
        """
        if not self.minio_client:
            raise Exception("MinIO клиент не инициализирован")
        
        try:
            # Конвертируем DataFrame в CSV в памяти
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=True)
            csv_buffer.seek(0)
            
            # Загружаем в MinIO
            csv_bytes = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
            self.minio_client.put_object(
                Bucket=bucket_name,
                Key=file_key,
                Body=csv_bytes.getvalue(),
                ContentType='text/csv'
            )
            self.log.info(f"Данные успешно сохранены в MinIO: {file_key}")
            return True
        except Exception as e:
            self.log.error(f"Ошибка сохранения CSV в MinIO: {e}")
            raise

    def preprocess_data(self, df):
        # Удаляем лишние пробелы в названиях столбцов
        df.columns = df.columns.str.strip()

        # Столбцы, которые хотим удалить (если есть)
        columns_to_drop_cat = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
        columns_to_drop = [
            'id','Total Fwd Packets', 'Flow IAT Mean', 'Fwd Packet Length Std', 'Bwd IAT Mean',
            'Bwd IAT Max', 'Fwd IAT Total', 'Active Max', 'Fwd IAT Min',
            'Fwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Total', 'Fwd PSH Flags', 'FIN Flag Count',
            'Active Min', 'Down/Up Ratio', 'Bwd IAT Min', 'Active Std', 'Fwd Packet Length Min',
            'SYN Flag Count', 'Active Mean', 'Idle Std', 'Bwd PSH Flags', 'Bwd URG Flags',
            'Fwd URG Flags', 'Fwd Avg Bytes/Bulk', 'RST Flag Count', 'CWE Flag Count',
            'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk',
            'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'ECE Flag Count'
        ]

        # Удаляем возможные дубликаты в списке колонок (сохраняя порядок)
        columns_to_drop = list(dict.fromkeys(columns_to_drop))

        # Создаём целевой столбец State: BENIGN -> 1, иначе 0
        if 'Label' not in df.columns:
            self.log.error("Column 'Label' not present in dataframe during preprocessing")
            raise KeyError("Label column missing")
        df['State'] = df['Label'].map(lambda a: 1 if a == 'BENIGN' else 0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Убедимся, что отступы ровные — строка ниже находится на том же уровне, что и предыдущие
        X = df.drop(columns=columns_to_drop_cat + columns_to_drop + ['State'], errors='ignore')
        y = df['State']
        return X, y


    def get_data(self) -> bool:
        '''
        Загрузка, предобработка и сохранение данных
        '''
        try:
            cfg_dir = os.path.dirname(self.config_path)

            # безопасно читаем train_file
            train_file_val = self.config.get('DATA', 'train_file', fallback=None)
            if train_file_val is None:
                self.log.error('train_file не задан в секции DATA')
                return False
            # убираем кавычки и пробелы
            train_file_val = train_file_val.strip().strip('"').strip("'")

            # Проверяем, доступен ли MinIO
            minio_config = self._get_minio_config()
            if self.minio_client and minio_config:
                # Загружаем данные из MinIO
                self.log.info("Используем MinIO для загрузки данных")
                train_df = self._load_csv_from_minio(minio_config['bucket_name'], train_file_val)
            else:
                # Загружаем из локального файла
                if not os.path.isabs(train_file_val):
                    train_file = os.path.normpath(os.path.join(cfg_dir, train_file_val))
                else:
                    train_file = os.path.normpath(train_file_val)

                if not os.path.isfile(train_file):
                    self.log.error(f"Train file not found: {train_file}")
                    return False

                train_df = pd.read_csv(train_file, encoding='latin1', low_memory=False)
            
            X_train, y_train = self.preprocess_data(train_df)
            
            # Сохранение предобработанных обучающих данных
            if self.minio_client and minio_config:
                # Сохраняем в MinIO
                self._save_csv_to_minio(X_train, minio_config['bucket_name'], "preprocessed_train_X.csv")
                self._save_csv_to_minio(y_train, minio_config['bucket_name'], "preprocessed_train_y.csv")
            else:
                # Сохраняем локально
                X_train.to_csv(self.train_path[0], index=True)
                y_train.to_csv(self.train_path[1], index=True)

            # Загрузка тестовых данных
            test_file_val = self.config.get('DATA', 'test_file', fallback=None)
            if not test_file_val:
                self.log.error('test_file не задан в секции DATA')
                return False

            test_file_val = test_file_val.strip().strip('"').strip("'")

            if self.minio_client and minio_config:
                # Загружаем данные из MinIO
                test_df = self._load_csv_from_minio(minio_config['bucket_name'], test_file_val)
            else:
                if not os.path.isabs(test_file_val):
                    test_path = os.path.normpath(os.path.join(cfg_dir, test_file_val))
                else:
                    test_path = os.path.normpath(test_file_val)

                if not os.path.isfile(test_path):
                    self.log.error(f"Test file not found: {test_path}")
                    return False

                test_df = pd.read_csv(test_path, encoding='latin1', low_memory=False)
            
            X_test, y_test = self.preprocess_data(test_df)

            # Сохранение предобработанных тестовых данных
            if self.minio_client and minio_config:
                # Сохраняем в MinIO
                self._save_csv_to_minio(X_test, minio_config['bucket_name'], "preprocessed_test_X.csv")
                self._save_csv_to_minio(y_test, minio_config['bucket_name'], "preprocessed_test_y.csv")
            else:
                # Сохраняем локально
                X_test.to_csv(self.test_path[0], index=True)
                y_test.to_csv(self.test_path[1], index=True)

            self.log.info("X and y data is ready")
            
            # Обновляем конфигурацию
            if self.minio_client and minio_config:
                # Если используем MinIO, сохраняем пути к MinIO
                self.config['PREPROCESSED_DATA'] = {
                    'X_train': f"minio://{minio_config['bucket_name']}/preprocessed_train_X.csv",
                    'y_train': f"minio://{minio_config['bucket_name']}/preprocessed_train_y.csv",
                    'X_test': f"minio://{minio_config['bucket_name']}/preprocessed_test_X.csv",
                    'y_test': f"minio://{minio_config['bucket_name']}/preprocessed_test_y.csv"
                }
            else:
                # Если используем локальные файлы
                self.config['PREPROCESSED_DATA'] = {
                    'X_train': self.train_path[0],
                    'y_train': self.train_path[1],
                    'X_test': self.test_path[0],
                    'y_test': self.test_path[1]
                }
            
            return True  # Все данные успешно обработаны
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            return False
        except Exception as e:
            self.log.error(f"Error in get_data: {str(e)}")
            self.log.debug(traceback.format_exc())
            return False

    def split_data(self) -> bool:
        '''
        Разбиваем данные на обучающую и тестовую выборку и сохраняем
        '''
        if not self.get_data():
            # не делаем sys.exit в библиотечном коде — вернём False, чтобы тесты могли обработать ошибку
            return False

        # Обновляем конфигурацию
        minio_config = self._get_minio_config()
        if self.minio_client and minio_config:
            self.config['PREPROCESSED_DATA'] = {
                'X_train': f"minio://{minio_config['bucket_name']}/preprocessed_train_X.csv",
                'y_train': f"minio://{minio_config['bucket_name']}/preprocessed_train_y.csv",
                'X_test': f"minio://{minio_config['bucket_name']}/preprocessed_test_X.csv",
                'y_test': f"minio://{minio_config['bucket_name']}/preprocessed_test_y.csv"
            }
        else:
            self.config['PREPROCESSED_DATA'] = {
                'X_train': self.train_path[0],
                'y_train': self.train_path[1],
                'X_test': self.test_path[0],
                'y_test': self.test_path[1]
            }
        
        self.log.info("Train and test data is ready")

        # Запишем обновлённый конфиг в тот же файл, откуда читали
        try:
            with open(self.config_path, 'w') as configfile:
                self.config.write(configfile)
        except Exception:
            self.log.warning(f"Не удалось записать конфиг по пути {self.config_path}")

        return True

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        df = df.reset_index(drop=True)
        
        # Проверяем, является ли путь MinIO-путем
        if path.startswith("minio://"):
            try:
                # Извлекаем bucket_name и file_key из пути
                parts = path.replace("minio://", "").split("/", 1)
                if len(parts) != 2:
                    raise ValueError(f"Неверный формат MinIO пути: {path}")
                
                bucket_name, file_key = parts
                
                # Сохраняем в MinIO
                if self.minio_client:
                    return self._save_csv_to_minio(df, bucket_name, file_key)
                else:
                    raise Exception("MinIO клиент не инициализирован")
            except Exception as e:
                self.log.error(f"Ошибка сохранения в MinIO: {e}")
                return False
        else:
            # Убедимся, что директория для пути существует
            dirn = os.path.dirname(path)
            if dirn:
                try:
                    os.makedirs(dirn, exist_ok=True)
                except Exception:
                    pass
            df.to_csv(path, index=True)
            self.log.info(f'{path} is saved')
            return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()