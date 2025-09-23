# src/kafka_consumer.py
import os, json, threading, time
from kafka import KafkaConsumer
from typing import Callable
from src.secrets import get_redis_config
import redis

KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC','predictions')
GROUP_ID = os.getenv('KAFKA_CONSUMER_GROUP','model-consumers')

_stop = threading.Event()

def default_handler(msg: dict):
    print("Consumed:", msg)
    # пример: сохранить результат в redis
    cfg = get_redis_config()
    r = redis.Redis(
        host=cfg['REDIS_HOST'],
        port=int(cfg['REDIS_PORT']),
        password=cfg['REDIS_PASSWORD'],
        db=int(cfg['REDIS_DB']),
        decode_responses=True
    )
    # сохраняем с key = request_id или timestamp
    key = "prediction:" + str(msg.get('meta', {}).get('request_id', time.time()))
    r.set(key, json.dumps(msg))

def loop(handler: Callable[[dict], None] = default_handler):
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        consumer_timeout_ms=1000
    )
    try:
        while not _stop.is_set():
            for rec in consumer:
                try:
                    handler(rec.value)
                except Exception as e:
                    print("Handler error:", e)
                if _stop.is_set():
                    break
            time.sleep(0.5)
    finally:
        consumer.close()

def start_in_thread(handler: Callable[[dict], None] = default_handler):
    t = threading.Thread(target=loop, args=(handler,), daemon=True)
    t.start()
    return t

def stop():
    _stop.set()
