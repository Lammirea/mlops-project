# src/kafka_consumer.py
import os, json, threading, time
from kafka import KafkaConsumer
from typing import Callable
from src.secret import get_postgres_config
import psycopg2
from psycopg2 import sql

KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC','predictions')
GROUP_ID = os.getenv('KAFKA_CONSUMER_GROUP','model-consumers')

_stop = threading.Event()

def default_handler(msg: dict):
    print("Consumed:", msg)
    # пример: сохранить результат в postgresql
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
        
        # Создаем таблицу для хранения предсказаний, если её нет
        create_table_query = """
        CREATE TABLE IF NOT EXISTS kafka_predictions (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(255),
            prediction_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_query)
        
        # сохраняем с key = request_id или timestamp
        request_id = str(msg.get('meta', {}).get('request_id', time.time()))
        prediction_data = json.dumps(msg)
        
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