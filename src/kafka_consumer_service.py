# src/kafka_consumer_service.py
import signal
import sys
from src.kafka_consumer import loop, stop

def _handle_signal(sig, frame):
    print(f"Signal {sig} received, stopping consumer...")
    stop()
    # даём loop время корректно завершиться
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    print("Starting kafka consumer service (blocking)...")
    try:
        loop()  # блокирующий вызов
    except SystemExit:
        pass
    except Exception as e:
        print("Consumer service crashed with exception:", e)
    finally:
        print("Consumer service stopped")

if __name__ == "__main__":
    main()
