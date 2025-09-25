# src/kafka_consumer.py
import os
import json
import threading
import time
from typing import Callable, Optional

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import sys

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pg_conn import get_engine, Base, InferenceResult
from sqlalchemy.orm import sessionmaker

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "send_prediction")
GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP", "model_consumer_group")

_stop = threading.Event()

# Инициализация БД (создание таблиц при старте модуля)
engine = get_engine()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def default_handler(msg: dict):
    """
    msg ожидается в формате похожем на code2: {"input": ..., "prediction": ...}
    Но если структура другая — сохраняем input_data как json всего сообщения.
    """
    session = Session()
    try:
        # Попробуем получить поля как в code2, иначе сохраняем весь объект как input
        input_part = msg.get("input", msg)
        prediction_raw = msg.get("prediction")

        input_json = json.dumps(input_part, default=str, ensure_ascii=False)
        prediction_val: Optional[float] = None
        if prediction_raw is not None:
            try:
                prediction_val = float(prediction_raw)
            except (ValueError, TypeError):
                # если не удалось привести к float — сохраняем None и логируем
                print(f"Warning: prediction value can't be converted to float: {prediction_raw}")

        record = InferenceResult(
            input_data=input_json,
            prediction=prediction_val
        )
        session.add(record)
        session.commit()
        print(f"Saved to DB: id={record.id}, prediction={record.prediction}")
    except Exception as e:
        session.rollback()
        print("Error saving record to DB:", e)
    finally:
        session.close()


def _make_consumer_with_retry():
    backoff = 1.0
    max_backoff = 30.0
    while not _stop.is_set():
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=[KAFKA_BOOTSTRAP],
                group_id=GROUP_ID,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000,
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
    Блокирующий loop с переподключением. Запускайте в отдельном процессе/потоке.
    """
    while not _stop.is_set():
        consumer = None
        try:
            consumer = _make_consumer_with_retry()
            for rec in consumer:
                if _stop.is_set():
                    break
                try:
                    if rec is None or rec.value is None:
                        continue
                    handler(rec.value)
                except Exception as e:
                    print("Handler error:", e)
        except Exception as e:
            print("Unexpected error in consumer loop:", e)
            time.sleep(2)
        finally:
            try:
                if consumer is not None:
                    consumer.close()
            except Exception:
                pass
        time.sleep(1)


def start_in_thread(handler: Callable[[dict], None] = default_handler):
    t = threading.Thread(target=loop, args=(handler,), daemon=True)
    t.start()
    return t


def stop():
    _stop.set()

if __name__ == "__main__":
    try:
        loop()
    except KeyboardInterrupt:
        stop()
