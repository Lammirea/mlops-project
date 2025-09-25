# MLOps prooject

The executed branch of this project is "develop". If everything gonna be alright i merge "develop" and "main".

In this oroject i train three ML models on CICIDS2017 dataset for attack type classification.

## Data

The data is stored in minio, which runs alongside the other services of the project.

## Arcitecture

- Vault: Ansible Vault for secret management
- Postgres: PostgreSQL database for storing predictions
- App: The machine learning model API

## Project structure

MLOPS_PROJECT/
├── data /                          # Папка с данными для обучения и тестирования модели (датасет CICIDS2017)
├── DevOps/
│   └── Windows/
│       ├── CI/                     # Скрипт CI для Jenkins
|       └── CD/                     # Скрипт для CD Jenkins                  
├── experiments/                     # Эксперименты с моделями ML
│   ├── d_tree.sav
│   ├── decision_tree_model.sav
│   ├── gnb.sav
│   ├── log_reg.sav
│   ├── preprocessor.sav
│   └── rand_forest.sav
├── infra/
│   └── ansible/                     # Инфраструктура через Ansible
│       ├── group_vars/
│       ├── playbooks/
│       ├── .vault_pass.txt          # Пароль для Ansible Vault (нет в репозитории!)
│       └── ansible.cfg
├── notebooks/                       # Jupyter-ноутбуки для анализа
│   └── network-traffic-analysis.ipynb
├── src/                             # Основной код приложения
│   ├── tests/                   # функциональные тесты
│   ├── unit_tests/              # Юнит-тесты
│   ├── __init__.py
│   ├── app.py                       # API
│   ├── kafka_consumer_service.py    
│   ├── kafka_consumer.py            
│   ├── kafka_producer.py            
│   ├── logfile.log                  # Лог-файл приложения
│   ├── logger.py                    # Настройка логирования
│   ├── predict.py                   # Скрипт предсказания
│   ├── preprocessor.py              # Предобработка данных
│   ├── secret.py                    # Хранение секретов (нет в репозитории!)
│   └── train.py                     # Обучение модели
├── .env                             # Переменные окружения
├── .env.vault                       # Зашифрованные переменные (Ansible Vault)
├── .gitattributes
├── .gitignore
├── vault-key.txt                    # Ключ для расшифровки Vault (нет в репозитории!)
├── bash.bat                         # Батник для запуска на Windows
├── config.ini                       # Конфигурационный файл
└── docker-compose.yml               # Docker Compose для локального развертывания

## API Endpoints

- /predict: Make a prediction based on a review
- /train: Make a train of choosen ML model
- /health: Fast health check of API

## How To Start API

```
curl http://localhost:8090/health
```
