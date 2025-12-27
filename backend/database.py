import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

# Default to nice local postgres url or sqlite for quick testing if env not set
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql://user:password@localhost:5432/fakenews_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    input_text_hash = Column(String, index=True)  # Sha1 of input text
    input_text_preview = Column(String)  # First 100 chars
    label = Column(String)
    probability = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, index=True)
    user_label = Column(String) # "REAL" or "FAKE" or "UNCERTAIN"
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # In production, use Alembic. For now, create all.
    Base.metadata.create_all(bind=engine)
