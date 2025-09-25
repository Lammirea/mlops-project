# src/main_app.py
import os
import json
import configparser
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import MultiModel
from src.predict import Predictor
from src.logger import Logger
from src.kafka_producer import send_message
from src.kafka_consumer import start_in_thread as start_kafka_consumer, stop as stop_kafka_consumer

# pg_conn provides SQLAlchemy engine, session and model
from src.pg_conn import get_engine, get_session, init_db, InferenceResult

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Конфигурация
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "send_prediction")

# Логгер
custom_logger_instance = Logger(show=True)
logger = custom_logger_instance.get_logger("AppLogger")

# Глобальные ресурсы приложения (устанавливаются в lifespan)
engine = None
consumer_thread = None

# SQL for cache table
CREATE_CACHE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(255) UNIQUE,
    cache_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
)
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for FastAPI: инициализация БД, кеш-таблицы и запуска consumer.
    """
    global engine, consumer_thread
    # Init DB models (eval_results)
    try:
        init_db()
        engine = get_engine()
        logger.info("Database initialized via init_db()")
    except Exception as e:
        logger.warning(f"Failed to initialize DB on startup: {e}")
        engine = None

    # Create cache table if engine available
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text(CREATE_CACHE_TABLE_SQL))
                logger.info("prediction_cache table ensured")
        except Exception as e:
            logger.warning(f"Failed to create/ensure prediction_cache table: {e}")

    # Start kafka consumer thread (best-effort)
    try:
        logger.info("Starting Kafka consumer thread")
        consumer_thread = start_kafka_consumer()
    except Exception as e:
        logger.warning(f"Failed to start Kafka consumer: {e}")
        consumer_thread = None

    yield

    # Shutdown: stop consumer and dispose engine if possible
    try:
        logger.info("Stopping Kafka consumer")
        stop_kafka_consumer()
    except Exception as e:
        logger.warning(f"Error stopping Kafka consumer: {e}")

    try:
        if engine is not None:
            engine.dispose()
    except Exception as e:
        logger.debug(f"Error disposing engine: {e}")

app = FastAPI(lifespan=lifespan)


def get_from_cache(cache_key: str):
    """
    Read cached JSON value from prediction_cache if present and not expired.
    Returns Python object or None.
    """
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT cache_value FROM prediction_cache WHERE cache_key = :k AND (expires_at IS NULL OR expires_at > NOW())"),
                {"k": cache_key}
            ).fetchone()
            if result and result[0]:
                return json.loads(result[0])
            return None
    except Exception as e:
        logger.warning(f"Error reading from cache: {e}")
        return None


def set_to_cache(cache_key: str, value: dict, expire_minutes: int = 60):
    """
    Write JSON-serializable `value` into prediction_cache with expiry.
    """
    if engine is None:
        return
    try:
        json_value = json.dumps(value, default=str)
        with engine.begin() as conn:
            # Upsert using Postgres ON CONFLICT
            conn.execute(
                text("""
                INSERT INTO prediction_cache (cache_key, cache_value, expires_at)
                VALUES (:k, :v, NOW() + INTERVAL ':m minutes')
                ON CONFLICT (cache_key) DO UPDATE
                SET cache_value = EXCLUDED.cache_value, expires_at = EXCLUDED.expires_at
                """),
                {"k": cache_key, "v": json_value, "m": str(expire_minutes)}
            )
    except Exception as e:
        logger.warning(f"Error writing to cache: {e}")


@app.get("/health")
async def health_check():
    """
    Health-check. Проверяем DB доступность и состояние kafka consumer.
    """
    status = {"status": "healthy", "postgres": "unknown", "kafka_consumer": "unknown"}

    # DB check
    try:
        if engine is not None:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            status["postgres"] = "connected"
        else:
            status["postgres"] = "not configured"
    except Exception as e:
        status["postgres"] = f"error: {e}"

    # Kafka consumer thread check
    try:
        if consumer_thread is not None and consumer_thread.is_alive():
            status["kafka_consumer"] = "running"
        else:
            status["kafka_consumer"] = "not running"
    except Exception:
        status["kafka_consumer"] = "unknown"

    overall = "healthy" if status["postgres"] in ["connected", "not configured"] else "unhealthy"
    return {"status": overall, "components": status}


@app.post("/train/")
async def train_model(
    background_tasks: BackgroundTasks,
    model_type: str = "d_tree",
    use_config: bool = True,
    save_model: bool = True,
    solver: str = "lbfgs",
    max_iter: int = 100,
    n_estimators: int = 100,
    criterion: str = "entropy",
    max_depth: int = 10,
    min_samples_split: int = 2,
    predict_flag: bool = False
):
    """
    Train model and (best-effort) send notification to Kafka via background task.
    """
    try:
        multi_model = MultiModel()

        if model_type == "log_reg":
            result = multi_model.log_reg(
                use_config=use_config, solver=solver, max_iter=max_iter, predict=predict_flag, save=save_model
            )
        elif model_type == "rand_forest":
            result = multi_model.rand_forest(
                use_config=use_config, n_estimators=n_estimators, criterion=criterion, predict=predict_flag, save=save_model
            )
        elif model_type == "d_tree":
            result = multi_model.d_tree(
                use_config=use_config, max_depth=max_depth, min_samples_split=min_samples_split, predict=predict_flag, save=save_model
            )
        elif model_type == "gnb":
            result = multi_model.gnb(predict=predict_flag, save=save_model)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

        payload = {"event": "model_trained", "model_type": model_type, "result": result, "saved": save_model}
        try:
            # send_message(topic, payload) as in code2
            background_tasks.add_task(send_message, KAFKA_TOPIC, payload)
        except Exception as e:
            logger.warning(f"Failed to schedule send_message to Kafka: {e}")

        return {"model_trained": result, "model_type": model_type, "model_saved": save_model}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
async def predict_model(
    background_tasks: BackgroundTasks,
    mode: str = "smoke",
    file: UploadFile = None
):
    """
    Prediction endpoint. Try cache first; otherwise predict and save to cache + notify Kafka.
    """
    cache_key = f"predict:{mode}"

    # Try cache
    try:
        cached = get_from_cache(cache_key)
        if cached:
            return {"from_cache": True, **cached}
    except Exception as e:
        logger.warning(f"Cache read error: {e}")

    try:
        predictor = Predictor()

        if mode == "upload":
            if file is None:
                raise HTTPException(status_code=400, detail="File required for mode 'upload'")
            file_contents = await file.read()
            result = predictor.predict_upload(file_contents)
        elif mode == "smoke":
            result = predictor.predict()
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'smoke' or 'upload'")

        safe_result = jsonable_encoder(result)

        # Best-effort: store in cache
        try:
            set_to_cache(cache_key, safe_result)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

        # Send to Kafka in background
        try:
            payload = {"event": "prediction", "mode": mode, "prediction": safe_result}
            background_tasks.add_task(send_message, KAFKA_TOPIC, payload)
        except Exception as e:
            logger.warning(f"Failed to schedule send_message to Kafka: {e}")

        # Optionally also persist to eval_results table (best-effort) using InferenceResult
        try:
            sess = get_session()
            rec = InferenceResult(input_data=json.dumps(safe_result, ensure_ascii=False), prediction=None)
            sess.add(rec)
            sess.commit()
            sess.close()
        except Exception as e:
            logger.debug(f"Failed to persist prediction to eval_results: {e}")

        return safe_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/receive_result")
async def receive_result(payload: dict):
    """
    Endpoint to accept results pushed by external consumer/services.
    Optionally save to eval_results table.
    """
    try:
        logger.info(f"Received result via /receive_result: {payload}")
        # Best-effort persist
        try:
            sess = get_session()
            inp = payload.get("input") or payload
            pred = None
            # try to extract numeric prediction if present
            if isinstance(payload.get("prediction"), (int, float, str)):
                try:
                    pred = float(payload.get("prediction"))
                except Exception:
                    pred = None
            rec = InferenceResult(input_data=json.dumps(inp, default=str, ensure_ascii=False), prediction=pred)
            sess.add(rec)
            sess.commit()
            sess.close()
        except Exception as e:
            logger.warning(f"Failed to save received payload to eval_results: {e}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error in receive_result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # If running directly, ensure DB tables exist
    try:
        init_db()
    except Exception as e:
        print(f"init_db failed: {e}")

    # run via uvicorn externally as needed
