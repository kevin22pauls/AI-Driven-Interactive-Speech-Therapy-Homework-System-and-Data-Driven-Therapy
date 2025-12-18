from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from config import DB_URL

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
metadata = MetaData()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
