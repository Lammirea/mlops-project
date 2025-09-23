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

def get_redis_config() -> Dict[str, Optional[str]]:
    """
    Priority:
     1) big_data_lab_three/.env (created by ansible-vault decrypt)
     2) /run/secrets/* files (docker secrets)
     3) environment variables
    """
    # 1) repo .env (path relative to container working dir)
    possible_paths = [
        os.path.join(os.getcwd(), 'big_data_lab_three', '.env'),
        os.path.join(os.getcwd(), '.env'),
        '/etc/secrets/redis_config.json',
        '/run/secrets/redis_password'  # if separate
    ]
    env_map = {}
    for p in possible_paths:
        if os.path.exists(p):
            env_map.update(_read_env_file(p))
            break

    # 2) file-based secrets (individual)
    # if separate files exist, read them
    for secret_name in ('REDIS_HOST','REDIS_PORT','REDIS_PASSWORD','REDIS_DB'):
        path = f'/run/secrets/{secret_name.lower()}'
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                env_map[secret_name] = f.read().strip()

    # 3) fallback to environment variables if missing
    cfg = {
        'REDIS_HOST': env_map.get('REDIS_HOST') or os.getenv('REDIS_HOST','localhost'),
        'REDIS_PORT': env_map.get('REDIS_PORT') or os.getenv('REDIS_PORT','6379'),
        'REDIS_PASSWORD': env_map.get('REDIS_PASSWORD') or os.getenv('REDIS_PASSWORD'),
        'REDIS_DB': env_map.get('REDIS_DB') or os.getenv('REDIS_DB','0')
    }
    return cfg
