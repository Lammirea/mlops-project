from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
import uvicorn
import os
import configparser
import json
import redis
from src.train import MultiModel
from src.predict import Predictor
from fastapi.encoders import jsonable_encoder
from src.logger import Logger
from src.kafka_producer import send_prediction
from src.secret import get_redis_config
from src.kafka_consumer import start_in_thread as start_kafka_consumer, stop as stop_kafka_consumer

from contextlib import asynccontextmanager

# Инициализация кастомного логгера
custom_logger_instance = Logger(show=True)  # show=True — вывод в консоль
logger = custom_logger_instance.get_logger("AppLogger")  # Получаем логгер

# Глобальная переменная для redis клиента; будет инициализирована в startup
redis_client = None
consumer_thread = None

def create_redis_client_from_config():
    """
    Создаём redis клиент на основе get_redis_config() (чтобы читать .env, docker secrets или env vars).
    Возвращаем объект redis.Redis или None (если подключение не удалось).
    """
    cfg = get_redis_config()
    host = cfg.get('REDIS_HOST', 'localhost')
    port = int(cfg.get('REDIS_PORT', 6379) or 6379)
    password = cfg.get('REDIS_PASSWORD', None)
    db = int(cfg.get('REDIS_DB', 0) or 0)

    try:
        rc = redis.Redis(host=host, port=port, password=password, db=db, decode_responses=True)
        # проверка подключения
        rc.ping()
        logger.info(f"Connected to Redis at {host}:{port} db={db}")
        return rc
    except Exception as e:
        logger.warning(f"Redis not available at startup (best-effort): {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, consumer_thread

    # Startup
    try:
        redis_client = create_redis_client_from_config()
    except Exception as e:
        logger.warning(f"Ошибка при создании Redis клиента: {e}")
        redis_client = None

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
        if redis_client is not None:
            redis_client.close()
    except Exception as e:
        logger.debug(f"Ошибка при закрытии Redis клиента: {e}")

app = FastAPI(lifespan=lifespan)

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
    Эндпоинт предсказаний. Сначала пробуем взять из Redis cache, иначе выполняем Predict.
    После получения результата: сохраняем в Redis (best-effort) и отправляем сообщение в Kafka (в фоне).
    """
    cache_key = f"predict:{mode}"

    # Попытка чтения из кеша
    try:
        if redis_client is not None:
            raw = redis_client.get(cache_key)
            if raw:
                try:
                    parsed = json.loads(raw)
                    return {"from_cache": True, **parsed}
                except Exception:
                    logger.warning("Не удалось распарсить данные из Redis, игнорируем кэш")
    except redis.exceptions.RedisError as re:
        logger.warning(f"Redis error while checking cache (treat as cache miss): {re}")
    except Exception as e:
        logger.warning(f"Unexpected error during Redis cache check: {e}")

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

        # Best-effort: сохранить в Redis
        try:
            if redis_client is not None:
                try:
                    redis_client.set(cache_key, json.dumps(safe_result))
                except Exception as e:
                    logger.warning(f"Unexpected error while writing to Redis cache: {e}")
        except Exception as e:
            logger.warning(f"Redis error while storing cache (ignored): {e}")

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
        # При необходимости можно сохранить payload в Redis, БД и т.д.
        try:
            if redis_client is not None:
                key = f"received:{payload.get('meta', {}).get('request_id', '')}"
                redis_client.set(key, json.dumps(payload))
        except Exception as e:
            logger.warning(f"Не удалось записать полученный результат в Redis: {e}")
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
