# src/secrets.py
import os
from typing import Dict, Optional

def _read_env_file(path: str) -> Dict[str,str]:
    env = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k,v = line.split('=',1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def get_postgres_config() -> Dict[str, Optional[str]]:
    """
    Priority:
     1) mlops_project/.env (created by ansible-vault decrypt)
     2) /run/secrets/* files (docker secrets)
     3) environment variables
    """
    # 1) repo .env (path relative to container working dir)
    possible_paths = [
        os.path.join(os.getcwd(), 'mlops_project', '.env'),
        os.path.join(os.getcwd(), '.env'),
        '/etc/secrets/postgres_config.json',
        '/run/secrets/postgres_password'  # if separate
    ]
    env_map = {}
    for p in possible_paths:
        if os.path.exists(p):
            env_map.update(_read_env_file(p))
            break

    # 2) file-based secrets (individual)
    # if separate files exist, read them
    for secret_name in ('POSTGRES_HOST','POSTGRES_PORT','POSTGRES_DB','POSTGRES_USER','POSTGRES_PASSWORD'):
        path = f'/run/secrets/{secret_name.lower()}'
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                env_map[secret_name] = f.read().strip()

    # 3) fallback to environment variables if missing
    cfg = {
        'POSTGRES_HOST': env_map.get('POSTGRES_HOST') or os.getenv('POSTGRES_HOST','localhost'),
        'POSTGRES_PORT': env_map.get('POSTGRES_PORT') or os.getenv('POSTGRES_PORT','5432'),
        'POSTGRES_DB': env_map.get('POSTGRES_DB') or os.getenv('POSTGRES_DB','postgres'),
        'POSTGRES_USER': env_map.get('POSTGRES_USER') or os.getenv('POSTGRES_USER','postgres'),
        'POSTGRES_PASSWORD': env_map.get('POSTGRES_PASSWORD') or os.getenv('POSTGRES_PASSWORD')
    }
    return cfg

# Алиас для обратной совместимости (опционально)
def get_redis_config():
    """Deprecated: use get_postgres_config instead"""
    import warnings
    warnings.warn("get_redis_config is deprecated, use get_postgres_config", DeprecationWarning)
    return get_postgres_config()