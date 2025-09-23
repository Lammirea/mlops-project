import os
import unittest
import redis
import sys

try:
    import fakeredis
except ImportError:
    fakeredis = None

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(current_dir, "../..", "config.ini"))

from src.secret import get_redis_config

class TestRedisIntegration(unittest.TestCase):
    def setUp(self):
        cfg = get_redis_config()
        self.redis_host = cfg.get('REDIS_HOST','localhost')
        self.redis_port = int(cfg.get('REDIS_PORT',6379))
        self.redis_password = cfg.get('REDIS_PASSWORD', None)
        self.redis_db = int(cfg.get('REDIS_DB',0))

        real_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            db=self.redis_db,
            decode_responses=True
        )
        try:
            if real_client.ping():
                self.redis_client = real_client
                return
        except redis.exceptions.ConnectionError:
            pass

        if fakeredis is not None:
            self.redis_client = fakeredis.FakeStrictRedis(decode_responses=True)
        else:
            raise unittest.SkipTest("Redis недоступен и fakeredis не установлен")

    def test_redis_connection(self):
        self.assertTrue(self.redis_client.ping())

    def test_data_persistence(self):
        key='test_key'; val='test_value'
        self.redis_client.delete(key)
        self.redis_client.set(key,val)
        self.assertEqual(self.redis_client.get(key), val)
