# src/logger.py
import logging
import os
import sys
import warnings
import tempfile

warnings.filterwarnings("ignore")

class Logger:
    """
        Class for logging behaviour of data exporting - object of ExportingTool class
    """

    def __init__(self, show, enable=True) -> None:
        """
            Re-defined __init__ method which sets show parametr

        Args:
            show (bool): if set all logs will be shown in terminal
            enable (bool): allow adding handlers at all (useful for tests)
        """
        self.FORMATTER = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        # Prefer explicit env var, otherwise use system temp dir which is usually writable
        self.LOG_FILE = os.environ.get("LOG_FILE",
                                       os.path.join(tempfile.gettempdir(), "logfile.log"))
        self.show = show
        self.enable = enable

    def clear_log_file(self) -> None:
        """Очищает файл логов (если это возможно)."""
        try:
            if os.path.exists(self.LOG_FILE):
                # попытаться перезаписать файл (если есть права)
                with open(self.LOG_FILE, 'w'):
                    pass
        except (PermissionError, OSError):
            # не критично — просто пропускаем очистку, чтобы тесты не падали
            pass

    def get_console_handler(self) -> logging.StreamHandler:
        """
            Возвращает консольный handler для вывода в stdout
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.FORMATTER)
        return console_handler

    def get_file_handler(self):
        """
            Возвращает файловый handler, либо None если файл недоступен
        """
        try:
            # Попробовать создать директорию, если указана (например, если LOG_FILE="/var/log/.../logfile.log")
            directory = os.path.dirname(self.LOG_FILE)
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except (PermissionError, OSError):
                    # не можем создать директорию — будем считать, что файловый хендлер недоступен
                    return None

            file_handler = logging.FileHandler(self.LOG_FILE, mode='a')
            file_handler.setFormatter(self.FORMATTER)
            return file_handler
        except (PermissionError, OSError):
            # Не удалось открыть файл для записи — возвращаем None и позволяем коду продолжить работу без файла
            return None

    def get_logger(self, logger_name: str):
        """
            Создаёт и настраивает логгер с указанным именем.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        if self.enable:
            # если нужно — добавить консольный handler
            if self.show:
                logger.addHandler(self.get_console_handler())

            # попытаемся добавить файловый handler; если он не доступен, просто пропустим
            file_handler = self.get_file_handler()
            if file_handler:
                logger.addHandler(file_handler)
            else:
                # Чтобы не зависеть от наличия хендлеров и избежать ошибок, добавим NullHandler
                # (не выводит ничего, но предотвращает "No handler" ситуации)
                logger.addHandler(logging.NullHandler())

        logger.propagate = False
        return logger
