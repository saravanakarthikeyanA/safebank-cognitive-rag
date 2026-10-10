from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base

def get_engine(connection_string):
    # SQLite multithreading support for web apps
    connect_args = {}
    if connection_string.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    engine = create_engine(connection_string, connect_args=connect_args)
    return engine

def get_session_maker(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db(connection_string):
    """Initializes the database by creating all tables if they do not exist."""
    engine = get_engine(connection_string)
    Base.metadata.create_all(bind=engine)
    return engine
