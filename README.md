# MLOps prooject

The executed branch of this project is "develop". If everything gonna be alright i merge "develop" and "main".

In this oroject i train three ML models on CICIDS2017 dataset for attack type classification.

## Arcitecture

- Vault: Ansible Vault for secret management
- Postgres: PostgreSQL database for storing predictions
- App: The machine learning model API

## API Endpoints

- /predict: Make a prediction based on a review
- /train: Make a train of choosen ML model
- /healt: Fast health check of API

## CD

CD starts every monday at around 9am
