# src/kafka_producer.py
import os
import json
import threading
import time
from typing import Any, Optional

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable, KafkaError

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "send_prediction")  # тот же топик, что и consumer
# Управляем повторными попытками при отправке/подключении
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 30.0

# Глобальный переиспользуемый producer и блокировка для thread-safety
_producer_lock = threading.Lock()
_producer: Optional[KafkaProducer] = None
_producer_stop = threading.Event()


def _make_producer_with_retry():
    """
    Создаёт и возвращает KafkaProducer, выполняя экспоненциальный бэкофф при ошибках.
    Бросает RuntimeError если stop запрошен до создания producer.
    """
    backoff = _INITIAL_BACKOFF
    while not _producer_stop.is_set():
        try:
            p = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP],
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                # можно настроить дополнительные параметры (acks, retries) при необходимости
                retries=5,
            )
            print("Connected KafkaProducer to", KAFKA_BOOTSTRAP)
            return p
        except NoBrokersAvailable as e:
            print(f"Kafka broker not available ({e}). Retrying in {backoff:.1f}s...")
        except Exception as e:
            print(f"Error creating KafkaProducer: {e}. Retrying in {backoff:.1f}s...")
        time.sleep(backoff)
        backoff = min(_MAX_BACKOFF, backoff * 2)
    raise RuntimeError("Stop requested before Kafka producer could be created")


def get_producer():
    """
    Возвращает глобальный KafkaProducer, создавая его при необходимости.
    Переиспользуемый и потокобезопасный.
    """
    global _producer
    if _producer is not None:
        return _producer

    with _producer_lock:
        if _producer is None:
            _producer = _make_producer_with_retry()
    return _producer


def send_message(topic: str, payload: Any, sync: bool = True, timeout: float = 10.0):
    """
    Отправляет сообщение в указанный topic.
    Если sync=True — ждёт подтверждения отправки (future.get(timeout)).
    В случае ошибки — пробует пересоздать producer и повторить одну попытку.
    """
    producer = None
    try:
        producer = get_producer()
        future = producer.send(topic, payload)
        if sync:
            # дождаться результата (подтверждение отправки)
            result = future.get(timeout=timeout)
            # не возвращаем сложный объект, но логируем
            print(f"Message sent to {topic}, partition={result.partition}, offset={result.offset}")
            return result
        else:
            # асинхронная отправка — не ждём
            return None
    except (KafkaError, Exception) as e:
        print(f"Error sending message to Kafka: {e}. Attempting to recreate producer and retry once.")
        # попытка пересоздать producer и повторить один раз
        try:
            with _producer_lock:
                global _producer
                try:
                    if _producer is not None:
                        try:
                            _producer.close(timeout=5)
                        except Exception:
                            pass
                        _producer = None
                except NameError:
                    _producer = None
                _producer = _make_producer_with_retry()
            producer = _producer
            future = producer.send(topic, payload)
            if sync:
                result = future.get(timeout=timeout)
                print(f"Retry succeeded. Message sent to {topic}, partition={result.partition}, offset={result.offset}")
                return result
            else:
                return None
        except Exception as e2:
            print(f"Retry failed: {e2}")
            raise


def send_to_default(payload: Any, sync: bool = True, timeout: float = 10.0):
    """
    Удобная обёртка для отправки в TOPIC (тот же, что и consumer).
    """
    return send_message(TOPIC, payload, sync=sync, timeout=timeout)


def close_producer():
    """
    Закрыть глобальный producer (сбрасывает соединение).
    """
    global _producer
    with _producer_lock:
        if _producer is not None:
            try:
                _producer.flush()
                _producer.close(timeout=5)
                print("KafkaProducer closed")
            except Exception as e:
                print("Error closing KafkaProducer:", e)
            finally:
                _producer = None


def stop():
    """
    Запросить остановку (используется в тестах/graceful shutdown).
    """
    _producer_stop.set()
    close_producer()
