import configparser
import os
import pandas as pd
import numpy as np
import sys
import traceback

from src.logger import Logger

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

    def preprocess_data(self, df):
        # Удаляем лишние пробелы в названиях столбцов
        df.columns = df.columns.str.strip()

        # Столбцы, которые хотим удалить (если есть)
        columns_to_drop_cat = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
        columns_to_drop = [
            'Total Fwd Packets', 'Flow IAT Mean', 'Fwd Packet Length Std', 'Bwd IAT Mean',
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
            train_file_val = self.config.get('UTEST_DATA', 'train_file', fallback=None)
            if train_file_val is None:
                self.log.error('train_file не задан в секции UTEST_DATA')
                return False
            # убираем кавычки и пробелы
            train_file_val = train_file_val.strip().strip('"').strip("'")

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
            X_train.to_csv(self.train_path[0], index=True)
            y_train.to_csv(self.train_path[1], index=True)

            # Загрузка тестовых данных
            test_file_val = self.config.get('DATA', 'test_file', fallback=None)
            if not test_file_val:
                self.log.error('test_file не задан в секции DATA')
                return False

            test_file_val = test_file_val.strip().strip('"').strip("'")

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
            X_test.to_csv(self.test_path[0], index=True)
            y_test.to_csv(self.test_path[1], index=True)

            self.log.info("X and y data is ready")
            self.config['PREPROCESSED_DATA'] = {
                'X_train': self.train_path[0],
                'y_train': self.train_path[1],
                'X_test': self.test_path[0],
                'y_test': self.test_path[1]
            }
            return os.path.isfile(self.train_path[0]) and \
                   os.path.isfile(self.train_path[1]) and \
                   os.path.isfile(self.test_path[0]) and \
                   os.path.isfile(self.test_path[1])
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

        return os.path.isfile(self.train_path[0]) and \
               os.path.isfile(self.train_path[1]) and \
               os.path.isfile(self.test_path[0]) and \
               os.path.isfile(self.test_path[1])

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        df = df.reset_index(drop=True)
        # Убедимся, что директория для путьa существует
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