import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.preprocess import DataMaker
from src.logger import Logger
import sys
import tempfile

SHOW_LOG = True
IS_TEST_MODE = "pytest" in sys.modules or "unittest" in sys.modules

class MultiModel:
    def __init__(self):
        # Инициализация логгера и конфигурации
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        
        # Получаем директорию текущего файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Формируем путь на уровень выше (если config.ini в родительской папке)
        config_path = os.path.abspath(os.path.join(current_dir, "..", "config.ini"))
        self.config_path = config_path  # сохраняем, чтобы при записи знать куда писать
        
        print(f"Пытаемся загрузить конфиг из: {config_path}")
        
        if os.path.exists(config_path):
            self.config.read(config_path)
            self.log.info("Конфигурация успешно загружена")
        else:
            error_msg = f"Ошибка: файл {config_path} не найден"
            self.log.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Загрузка данных из файлов, указанных в config.ini
        train_path = os.path.normpath(os.path.join(os.getcwd(), self.config["UTEST_DATA"]["train_file"]))
        if not train_path:
            self.log.error('train_file не задан в секции UTEST_DATA')
            return False

        test_path = os.path.normpath(os.path.join(os.getcwd(), self.config["DATA"]["test_file"]))
        if not test_path:
            self.log.error('test_file не задан в секции DATA')
            return False

        train_df = pd.read_csv(train_path, encoding='latin1', low_memory=False)
        test_df = pd.read_csv(test_path, encoding='latin1', low_memory=False)
        
        # Предобработка данных
        data_preproc = DataMaker()
        self.X_train_raw, self.y_train = data_preproc.preprocess_data(train_df)
        self.X_test_raw, self.y_test = data_preproc.preprocess_data(test_df)
        self.feature_columns = list(self.X_train_raw.columns)
        
        # Создание pipeline для предобработки
        self.pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.X_train_scaled = self.pipeline.fit_transform(self.X_train_raw)
        self.X_test_scaled = self.pipeline.transform(self.X_test_raw)
        
        
        # Балансировка классов с помощью SMOTE
        smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train_scaled, self.y_train)
        
        # Путь для сохранения моделей и предобработчика
        # 1) попытаться взять из переменной окружения (удобно для CI)
        experiments_dir = os.environ.get("EXPERIMENTS_DIR")
        if experiments_dir:
            project_path = os.path.abspath(experiments_dir)
        else:
            project_path = os.path.join(os.getcwd(), "experiments")

        # Попытка создать директорию, при ошибке падаем на временную директорию
        try:
            if not os.path.exists(project_path):
                os.makedirs(project_path, exist_ok=True)
        except PermissionError:
            # fallback: системная временная папка (обычно доступна в контейнерах)
            fallback = os.path.join(tempfile.gettempdir(), "experiments")
            self.log.warning(f"Нет прав на создание {project_path}, переключаемся на {fallback}")
            project_path = fallback
            os.makedirs(project_path, exist_ok=True)

        self.project_path = project_path
        self.preprocessor_path = os.path.join(self.project_path, "preprocessor.sav")

        # Сохранение препроцессора с обработкой ошибок записи
        try:
            with open(self.preprocessor_path, "wb") as f:
                pickle.dump({'pipeline': self.pipeline, 'feature_columns': self.feature_columns}, f)
        except PermissionError:
            # если и тут PermissionError — используем ещё более безопасную временную папку
            fallback2 = os.path.join(tempfile.gettempdir(), "experiments")
            os.makedirs(fallback2, exist_ok=True)
            self.preprocessor_path = os.path.join(fallback2, "preprocessor.sav")
            with open(self.preprocessor_path, "wb") as f:
                pickle.dump({'pipeline': self.pipeline, 'feature_columns': self.feature_columns}, f)
            self.log.warning(f"Сохранили препроцессор во временную папку: {self.preprocessor_path}")

        # Пути для сохранения моделей
        self.log_reg_path = os.path.join(self.project_path, "log_reg.sav")
        self.rand_forest_path = os.path.join(self.project_path, "rand_forest.sav")
        self.gnb_path = os.path.join(self.project_path, "gnb.sav")
        self.d_tree_path = os.path.join(self.project_path, "d_tree.sav")
        
        self.log.info(f"MultiModel is ready. Models path: {self.project_path}")

    def log_reg(self, use_config: bool, solver="lbfgs", max_iter=100, predict=False, save=True):
        if use_config:
            solver = self.config["LOG_REG"].get("solver", solver)
            max_iter = self.config.getint("LOG_REG", "max_iter", fallback=max_iter)
        classifier = LogisticRegression(solver=solver, max_iter=max_iter)
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'solver': solver, 'max_iter': str(max_iter), 'path': self.log_reg_path}
        return self.save_model(classifier, self.log_reg_path, "LOG_REG", params, save)

    def rand_forest(self, use_config: bool, n_estimators=100, criterion="entropy", predict=False, save=True):
        if use_config:
            n_estimators = self.config.getint("RAND_FOREST", "n_estimators", fallback=n_estimators)
            criterion = self.config["RAND_FOREST"].get("criterion", criterion)
        classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'n_estimators': str(n_estimators), 'criterion': criterion, 'path': self.rand_forest_path}
        return self.save_model(classifier, self.rand_forest_path, "RAND_FOREST", params, save)

    def gnb(self, predict=False, save=True):
        classifier = GaussianNB()
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'path': self.gnb_path}
        return self.save_model(classifier, self.gnb_path, "GNB", params, save)

    def d_tree(self, use_config: bool, max_depth=10, min_samples_split=2, predict=False, save=True):
        if use_config:
            max_depth = self.config.getint("DECISION_TREE", "max_depth", fallback=max_depth)
            min_samples_split = self.config.getint("DECISION_TREE", "min_samples_split", fallback=min_samples_split)
        classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'max_depth': str(max_depth), 'min_samples_split': str(min_samples_split), 'path': self.d_tree_path}
        return self.save_model(classifier, self.d_tree_path, "DECISION_TREE", params, save)

    def save_model(self, classifier, path, section, params, save=True):
        # Сохранение модели и обновление конфигурации
        if save:
            self.config[section] = params
            # Пишем config.ini в тот путь, откуда он был загружен (self.config_path)
            try:
                with open(self.config_path, 'w') as configfile:
                    self.config.write(configfile)
            except Exception as e:
                # логируем, но не ломаем процесс — возможно тестовое окружение не разрешает запись
                self.log.warning(f"Не удалось перезаписать config.ini в {self.config_path}: {e}")

            # Пытаемся записать модель, при ошибке используем временную папку
            try:
                with open(path, 'wb') as f:
                    pickle.dump(classifier, f)
            except PermissionError:
                fallback = os.path.join(tempfile.gettempdir(), os.path.basename(path))
                with open(fallback, 'wb') as f:
                    pickle.dump(classifier, f)
                self.log.warning(f"Не получилось записать модель в {path}, сохранили в {fallback}")
                path = fallback

            self.log.info(f'{path} is saved')
            return os.path.isfile(path)
        else:
            self.log.info(f'Model training completed but not saved (save=False)')
            return True  # Возвращаем True, так как обучение успешно, просто не сохраняем
    
    def predict(self, model_name, test_type):
        """
        Выполняет предсказание для заданной модели и типа теста.
        """
        model_paths = {
            "log_reg": self.log_reg_path,
            "d_tree": self.d_tree_path,
            "gnb": self.gnb_path,
            "rand_forest": self.rand_forest_path
        }

        if model_name not in model_paths:
            raise ValueError(f"Unknown model: {model_name}")

        try:
            with open(model_paths[model_name], "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load model {model_name}: {e}")

        if test_type == "smoke":
            score = model.score(self.X_test_scaled, self.y_test)
            return {"test_score": score}
        else:
            raise NotImplementedError(f"Test type '{test_type}' is not implemented in this method.")
        
if __name__ == "__main__":
    multi_model = MultiModel()
    multi_model.log_reg(use_config=False, predict=True)