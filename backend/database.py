from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use SQLite for development
SQLALCHEMY_DATABASE_URL = "sqlite:///./anime_chatbot.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# --- THIS IS THE MISSING FUNCTION ---
# This is a dependency for our API routes. It creates a new database
# session for each request (like a login or registration) and ensures 
# the connection is properly closed afterward.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

