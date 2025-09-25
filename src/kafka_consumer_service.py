# src/kafka_consumer_service.py
import os
import sys

# Добавляем корень проекта в sys.path чтобы можно было писать "import src.*"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import signal
import sys as _sys
from src.kafka_consumer import loop, stop  # теперь import должен работать

def _handle_signal(sig, frame):
    print(f"Signal {sig} received, stopping consumer...")
    stop()
    _sys.exit(0)

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
