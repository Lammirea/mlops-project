from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
import uvicorn
import os
import configparser
import json
import psycopg2
from psycopg2 import sql
from src.train import MultiModel
from src.predict import Predictor
from fastapi.encoders import jsonable_encoder
from src.logger import Logger
from src.kafka_producer import send_prediction
from src.secret import get_postgres_config
from src.kafka_consumer import start_in_thread as start_kafka_consumer, stop as stop_kafka_consumer

from contextlib import asynccontextmanager

custom_logger_instance = Logger(show=True)
logger = custom_logger_instance.get_logger("AppLogger")

pg_connection = None
consumer_thread = None

def create_postgres_connection_from_config():
    """
    Создаём postgresql соединение на основе get_postgres_config().
    Возвращаем объект psycopg2.connection или None (если подключение не удалось).
    """
    cfg = get_postgres_config()
    host = cfg.get('POSTGRES_HOST', 'localhost')
    port = int(cfg.get('POSTGRES_PORT', 5432) or 5432)
    database = cfg.get('POSTGRES_DB', 'postgres')
    user = cfg.get('POSTGRES_USER', 'postgres')
    password = cfg.get('POSTGRES_PASSWORD', None)

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        # проверка подключения
        conn.cursor().execute("SELECT 1")
        logger.info(f"Connected to PostgreSQL at {host}:{port} db={database}")
        return conn
    except Exception as e:
        logger.warning(f"PostgreSQL not available at startup (best-effort): {e}")
        return None

def init_cache_table():
    """
    Создаём таблицу для кэширования результатов, если её нет
    """
    if pg_connection is None:
        return
    
    try:
        cursor = pg_connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS prediction_cache (
            id SERIAL PRIMARY KEY,
            cache_key VARCHAR(255) UNIQUE,
            cache_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
        """
        cursor.execute(create_table_query)
        pg_connection.commit()
        cursor.close()
        logger.info("Cache table initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize cache table: {e}")

def get_from_cache(cache_key: str):
    """
    Получить значение из PostgreSQL кэша
    """
    if pg_connection is None:
        return None
    
    try:
        cursor = pg_connection.cursor()
        select_query = "SELECT cache_value FROM prediction_cache WHERE cache_key = %s AND (expires_at IS NULL OR expires_at > NOW())"
        cursor.execute(select_query, (cache_key,))
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return json.loads(result[0])
        return None
    except Exception as e:
        logger.warning(f"Error reading from PostgreSQL cache: {e}")
        return None

def set_to_cache(cache_key: str, value: dict, expire_minutes: int = 60):
    """
    Сохранить значение в PostgreSQL кэш
    """
    if pg_connection is None:
        return
    
    try:
        cursor = pg_connection.cursor()
        # Удаляем старую запись, если есть
        delete_query = "DELETE FROM prediction_cache WHERE cache_key = %s"
        cursor.execute(delete_query, (cache_key,))
        
        # Вставляем новую запись
        insert_query = """
        INSERT INTO prediction_cache (cache_key, cache_value, expires_at) 
        VALUES (%s, %s, NOW() + INTERVAL '%s minutes')
        ON CONFLICT (cache_key) DO UPDATE 
        SET cache_value = EXCLUDED.cache_value, expires_at = EXCLUDED.expires_at
        """
        cursor.execute(insert_query, (cache_key, json.dumps(value), expire_minutes))
        pg_connection.commit()
        cursor.close()
    except Exception as e:
        logger.warning(f"Error writing to PostgreSQL cache: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pg_connection, consumer_thread

    # Startup
    try:
        pg_connection = create_postgres_connection_from_config()
        if pg_connection:
            init_cache_table()
    except Exception as e:
        logger.warning(f"Ошибка при создании PostgreSQL соединения: {e}")
        pg_connection = None

    try:
        logger.info("Starting Kafka consumer thread")
        consumer_thread = start_kafka_consumer()
    except Exception as e:
        logger.warning(f"Не удалось запустить Kafka consumer: {e}")

    yield  # Приложение запущено

    # Shutdown
    try:
        logger.info("Stopping Kafka consumer")
        stop_kafka_consumer()
    except Exception as e:
        logger.warning(f"Ошибка при остановке Kafka consumer: {e}")

    try:
        if pg_connection is not None:
            pg_connection.close()
    except Exception as e:
        logger.debug(f"Ошибка при закрытии PostgreSQL соединения: {e}")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """
    Проверка состояния API и подключений
    """
    status = {
        "status": "healthy",
        "postgres": "disconnected",
        "kafka_consumer": "unknown"
    }
    
    # Проверка PostgreSQL
    try:
        if pg_connection is not None:
            pg_connection.cursor().execute("SELECT 1")
            status["postgres"] = "connected"
        else:
            status["postgres"] = "not configured"
    except Exception as e:
        status["postgres"] = f"error: {str(e)}"
    
    # Проверка Kafka consumer (если возможно)
    try:
        if consumer_thread is not None and consumer_thread.is_alive():
            status["kafka_consumer"] = "running"
        else:
            status["kafka_consumer"] = "not running"
    except Exception:
        status["kafka_consumer"] = "unknown"
    
    overall_status = "healthy" if status["postgres"] in ["connected", "not configured"] else "unhealthy"
    return {
        "status": overall_status,
        "components": status
    }

@app.post("/train/")
async def train_model(
    background_tasks: BackgroundTasks,
    model_type: str = "d_tree",
    use_config: bool = True,
    save_model: bool = True,
    # Параметры для Logistic Regression
    solver: str = "lbfgs",
    max_iter: int = 100,
    # Параметры для Random Forest
    n_estimators: int = 100,
    criterion: str = "entropy",
    # Параметры для Decision Tree
    max_depth: int = 10,
    min_samples_split: int = 2,
    predict_flag: bool = False
):
    """
    Тренировка модели. По завершении отправляет сообщение в Kafka (producer) в фоне.
    """
    try:
        multi_model = MultiModel()

        if model_type == "log_reg":
            result = multi_model.log_reg(
                use_config=use_config,
                solver=solver,
                max_iter=max_iter,
                predict=predict_flag,
                save=save_model
            )
        elif model_type == "rand_forest":
            result = multi_model.rand_forest(
                use_config=use_config,
                n_estimators=n_estimators,
                criterion=criterion,
                predict=predict_flag,
                save=save_model
            )
        elif model_type == "d_tree":
            result = multi_model.d_tree(
                use_config=use_config,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                predict=predict_flag,
                save=save_model
            )
        elif model_type == "gnb":
            result = multi_model.gnb(predict=predict_flag, save=save_model)
        else:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип модели: {model_type}")

        # Подготовим уведомление для Kafka — не блокируем основной поток
        payload = {
            "event": "model_trained",
            "model_type": model_type,
            "result": result,
            "saved": save_model
        }
        try:
            background_tasks.add_task(send_prediction, payload, "predictions", False)
        except Exception as e:
            logger.warning(f"Не удалось поставить задачу отправки в Kafka: {e}")

        return {
            "model_trained": result,
            "model_type": model_type,
            "model_saved": save_model
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict_model(
    background_tasks: BackgroundTasks,
    mode: str = "smoke",
    file: UploadFile = None
):
    """
    Эндпоинт предсказаний. Сначала пробуем взять из PostgreSQL cache, иначе выполняем Predict.
    После получения результата: сохраняем в PostgreSQL (best-effort) и отправляем сообщение в Kafka (в фоне).
    """
    cache_key = f"predict:{mode}"

    # Попытка чтения из кэша
    try:
        cached_result = get_from_cache(cache_key)
        if cached_result:
            return {"from_cache": True, **cached_result}
    except Exception as e:
        logger.warning(f"Error during PostgreSQL cache check: {e}")

    try:
        predictor = Predictor()

        if mode == "upload":
            if file is None:
                raise HTTPException(status_code=400, detail="Файл не предоставлен для режима 'upload'")
            file_contents = await file.read()
            result = predictor.predict_upload(file_contents)
        elif mode == "smoke":
            result = predictor.predict()
        else:
            raise HTTPException(status_code=400, detail="Неверный режим. Используйте 'smoke' или 'upload'")

        # JSON-serializable версия
        safe_result = jsonable_encoder(result)

        # Best-effort: сохранить в PostgreSQL кэш
        try:
            set_to_cache(cache_key, safe_result)
        except Exception as e:
            logger.warning(f"Error while writing to PostgreSQL cache: {e}")

        # Отправим результат в Kafka в фоне
        try:
            payload = {
                "event": "prediction",
                "mode": mode,
                "prediction": safe_result
            }
            background_tasks.add_task(send_prediction, payload, "predictions", False)
        except Exception as e:
            logger.warning(f"Не удалось поставить задачу отправки в Kafka: {e}")

        return safe_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/receive_result")
async def receive_result(payload: dict):
    """
    Опциональный endpoint, если вы используете отдельный consumer/service и хотите,
    чтобы он присылал результаты обратно в web (например для подтверждения, логирования).
    """
    try:
        logger.info(f"Received result via /receive_result: {payload}")
        # При необходимости можно сохранить payload в PostgreSQL, БД и т.д.
        try:
            if pg_connection is not None:
                cache_key = f"received:{payload.get('meta', {}).get('request_id', '')}"
                set_to_cache(cache_key, payload)
        except Exception as e:
            logger.warning(f"Не удалось записать полученный результат в PostgreSQL: {e}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Ошибка в receive_result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '..', "config.ini")
    config.read(config_path, encoding="utf-8")
    try:
        host = config["FASTAPI"]["host"]
        port = config.getint("FASTAPI", "port")
    except KeyError:
        raise ValueError("В config.ini отсутствует секция [FASTAPI] или ключи host/port")
    uvicorn.run(app, host=host, port=port)