from sqlalchemy import create_engine, Column, Integer, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os


Base = declarative_base()

class InferenceResult(Base):
    __tablename__ = "eval_results"
    id = Column(Integer, primary_key=True)
    input_data = Column(Text)
    prediction = Column(Float)


def get_engine():
    user = os.getenv("POSTGRES_USER")
    pwd = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ml_predictions")
    url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)