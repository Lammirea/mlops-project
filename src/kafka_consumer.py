# src/kafka_consumer.py
import os
import json
import threading
import time
from typing import Callable
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from src.secret import get_postgres_config
import psycopg2

KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'predictions')
GROUP_ID = os.getenv('KAFKA_CONSUMER_GROUP', 'model-consumers')

_stop = threading.Event()

def default_handler(msg: dict):
    print("Consumed:", msg)
    cfg = get_postgres_config()
    try:
        conn = psycopg2.connect(
            host=cfg['POSTGRES_HOST'],
            port=int(cfg['POSTGRES_PORT']),
            database=cfg['POSTGRES_DB'],
            user=cfg['POSTGRES_USER'],
            password=cfg['POSTGRES_PASSWORD']
        )
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS kafka_predictions (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(255),
            prediction_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_query)
        request_id = str(msg.get('meta', {}).get('request_id', time.time()))
        prediction_data = json.dumps(msg, default=str, ensure_ascii=False)
        insert_query = """
        INSERT INTO kafka_predictions (request_id, prediction_data) 
        VALUES (%s, %s)
        """
        cursor.execute(insert_query, (request_id, prediction_data))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Prediction saved to PostgreSQL with request_id: {request_id}")
    except Exception as e:
        print(f"Error saving to PostgreSQL: {e}")

def _make_consumer_with_retry():
    backoff = 1.0
    max_backoff = 30.0
    while not _stop.is_set():
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=GROUP_ID,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000
            )
            print("Connected to Kafka broker(s) at", KAFKA_BOOTSTRAP)
            return consumer
        except NoBrokersAvailable as e:
            print(f"Kafka broker not available ({e}). Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)
        except Exception as e:
            print(f"Error creating KafkaConsumer: {e}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)
    raise RuntimeError("Stop requested before Kafka consumer could be created")

def loop(handler: Callable[[dict], None] = default_handler):
    """
    Blocking loop: поддерживает переподключение при падении/broker недоступен.
    Используйте этот блокирующий loop в отдельном процессе/контейнере.
    """
    while not _stop.is_set():
        consumer = None
        try:
            consumer = _make_consumer_with_retry()
            # Основной цикл чтения
            for rec in consumer:
                if _stop.is_set():
                    break
                try:
                    if rec is None or rec.value is None:
                        continue
                    handler(rec.value)
                except Exception as e:
                    print("Handler error:", e)
            # Если цикл for завершился (например, consumer timed out), просто продолжим и перезапустим
        except Exception as e:
            print("Unexpected error in consumer loop:", e)
            time.sleep(2)
        finally:
            try:
                if consumer is not None:
                    consumer.close()
            except Exception:
                pass
        # небольшая пауза перед попыткой пересоздать consumer
        time.sleep(1)

def start_in_thread(handler: Callable[[dict], None] = default_handler):
    t = threading.Thread(target=loop, args=(handler,), daemon=True)
    t.start()
    return t

def stop():
    _stop.set()