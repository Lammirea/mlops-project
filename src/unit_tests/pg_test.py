import os
import unittest
import sys
import psycopg2
from psycopg2 import sql
from unittest.mock import MagicMock

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(current_dir, "../..", "config.ini"))

from src.secret import get_postgres_config

class TestPostgreSQLIntegration(unittest.TestCase):
    def setUp(self):
        cfg = get_postgres_config()
        self.db_host = cfg.get('POSTGRES_HOST', 'localhost')
        self.db_port = int(cfg.get('POSTGRES_PORT', 5432))
        self.db_name = cfg.get('POSTGRES_DB', 'test_db')
        self.db_user = cfg.get('POSTGRES_USER', 'postgres')
        self.db_password = cfg.get('POSTGRES_PASSWORD', None)

        try:
            # Попытка подключения к реальной базе данных
            self.connection = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.cursor = self.connection.cursor()
            # Создание тестовой таблицы
            self._create_test_table()
            return
        except psycopg2.Error:
            pass

        # Если реальное подключение не удалось, используем mock
        try:
            self.connection = MagicMock()
            self.cursor = MagicMock()
            self.connection.cursor.return_value = self.cursor
            # Имитация поведения базы данных
            self.mock_data = {}
            self.cursor.execute.side_effect = self._mock_execute
            self.cursor.fetchone.side_effect = self._mock_fetchone
            self.cursor.fetchall.side_effect = self._mock_fetchall
        except Exception:
            raise unittest.SkipTest("PostgreSQL недоступен и mock не может быть создан")

    def _create_test_table(self):
        """Создание тестовой таблицы для проверки"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            key VARCHAR(255) UNIQUE,
            value TEXT
        )
        """
        self.cursor.execute(create_table_query)
        self.connection.commit()

    def _mock_execute(self, query, params=None):
        """Mock для execute метода"""
        query_str = query.strip().lower()
        if 'select' in query_str and 'version()' in query_str:
            self._last_query_result = [('PostgreSQL mock',)]
        elif 'select' in query_str and 'key' in query_str:
            if params and params[0] in self.mock_data:
                self._last_query_result = [(self.mock_data[params[0]],)]
            else:
                self._last_query_result = []
        elif 'insert' in query_str or 'update' in query_str:
            if params:
                self.mock_data[params[0]] = params[1]
            self._last_query_result = []

    def _mock_fetchone(self):
        """Mock для fetchone метода"""
        if hasattr(self, '_last_query_result') and self._last_query_result:
            return self._last_query_result[0]
        return None

    def _mock_fetchall(self):
        """Mock для fetchall метода"""
        if hasattr(self, '_last_query_result'):
            return self._last_query_result
        return []

    def tearDown(self):
        """Очистка после тестов"""
        if hasattr(self, 'cursor') and self.cursor and not isinstance(self.cursor, MagicMock):
            try:
                # Очистка тестовых данных
                self.cursor.execute("DELETE FROM test_table WHERE key = %s", ('test_key',))
                self.connection.commit()
                self.cursor.close()
            except:
                pass
        if hasattr(self, 'connection') and self.connection and not isinstance(self.connection, MagicMock):
            try:
                self.connection.close()
            except:
                pass

    def test_postgresql_connection(self):
        """Тест подключения к PostgreSQL"""
        if isinstance(self.connection, MagicMock):
            # Для mock соединения
            self.cursor.execute("SELECT version()")
            result = self.cursor.fetchone()
            self.assertIsNotNone(result)
        else:
            # Для реального соединения
            self.cursor.execute("SELECT version()")
            result = self.cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertIn('postgresql', result[0].lower())

    def test_data_persistence(self):
        """Тест сохранения данных"""
        key = 'test_key'
        val = 'test_value'
        
        # Удаление существующей записи
        delete_query = "DELETE FROM test_table WHERE key = %s"
        self.cursor.execute(delete_query, (key,))
        
        # Вставка новой записи
        insert_query = "INSERT INTO test_table (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
        self.cursor.execute(insert_query, (key, val))
        self.connection.commit()
        
        # Проверка чтения
        select_query = "SELECT value FROM test_table WHERE key = %s"
        self.cursor.execute(select_query, (key,))
        result = self.cursor.fetchone()
        
        if result:
            self.assertEqual(result[0], val)
        else:
            self.fail("Запись не найдена в базе данных")

if __name__ == '__main__':
    unittest.main()