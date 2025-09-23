# src/kafka_producer.py
import os, json
from kafka import KafkaProducer
from typing import Dict, Any

KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC','predictions')

def _json_serializer(data: Dict[str,Any]) -> bytes:
    return json.dumps(data, default=str, ensure_ascii=False).encode('utf-8')

def make_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=_json_serializer,
        acks='all',
        retries=5,
        linger_ms=5
    )

_producer = None
def get_producer():
    global _producer
    if _producer is None:
        _producer = make_producer()
    return _producer

def send_prediction(payload: Dict[str,Any], topic: str = KAFKA_TOPIC, sync: bool = False):
    p = get_producer()
    future = p.send(topic, payload)
    if sync:
        rec = future.get(timeout=10)
        return {"topic": rec.topic, "partition": rec.partition, "offset": rec.offset}
    else:
        def _on_error(err):
            print("Kafka send error:", err)
        future.add_errback(_on_error)
        return None
