import os
import uuid
from datetime import datetime
from threading import Lock

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    TIMESTAMP,
    func,
    ForeignKey,
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
# from dotenv import load_dotenv

# load_dotenv()

def _clean_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def _clean_port(raw_port: str) -> int:
    port_str = raw_port.strip().lstrip(":")
    try:
        return int(port_str)
    except ValueError as exc:
        raise RuntimeError(f"Invalid PGPORT value: {raw_port}") from exc


PG_USER = _clean_env("PGUSER")
PG_PASSWORD = _clean_env("PGPASSWORD")
PG_HOST = _clean_env("PGHOST")
PG_DATABASE = _clean_env("PGDATABASE")
PG_PORT = _clean_port(_clean_env("PGPORT"))

DATABASE_URL = (
    f"postgresql://{PG_USER}:{PG_PASSWORD}"
    f"@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

_db_initialized = False
_schema_lock = Lock()


def _ensure_schema() -> None:
    """Create tables lazily so requests never hit missing relations."""
    global _db_initialized
    if _db_initialized:
        return
    with _schema_lock:
        if _db_initialized:
            return
        Base.metadata.create_all(bind=engine)
        _ensure_ai_followup_column()
        _ensure_feedback_column()
        _ensure_user_feedback_rating_column()
        _db_initialized = True
# 🔥 NEW: Add feedback column safely
# ---------------------------------------------------------------------------
def _ensure_feedback_column() -> None:
    """Adds the feedback column to turns table ONLY IF missing."""
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("turns")}

    if "feedback" not in columns:
        print("⚠ Adding missing `feedback` column to turns table...")
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE turns ADD COLUMN feedback VARCHAR(20);")
                )
            print("✓ `feedback` column added successfully.")
        except Exception as e:
            print("⚠ Failed to add feedback column:", e)
    else:
        print("✓ `feedback` column already exists — skipping.")


def _ensure_ai_followup_column() -> None:
    """Add the ai_followup column and ensure it is nullable."""
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("turns")}
    ai_col = columns.get("ai_followup")
    if ai_col:
        if not ai_col.get("nullable", True):
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE turns ALTER COLUMN ai_followup DROP NOT NULL"))
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE turns ADD COLUMN ai_followup TEXT"))

def _ensure_user_feedback_rating_column() -> None:
    """Adds rating column to user_feedback table ONLY IF missing."""
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("user_feedback")}

    if "rating" not in columns:
        print("⚠ Adding missing `rating` column to user_feedback table...")
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE user_feedback "
                        "ADD COLUMN rating INTEGER NOT NULL DEFAULT 5;"
                    )
                )
            print("✓ `rating` column added successfully.")
        except Exception as e:
            print("⚠ Failed to add rating column:", e)
    else:
        print("✓ `rating` column already exists — skipping.")


class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, unique=True, index=True, nullable=False)
    url = Column(String)
    transcript = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

class Thread(Base):
    __tablename__ = "threads"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    last_active = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    turns = relationship("Turn", back_populates="thread", cascade="all, delete-orphan")

class Turn(Base):
    __tablename__ = "turns"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, index=True)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False, index=True)
    turn_index = Column(Integer, nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_message = Column(Text, nullable=False)
    youtube_suggestions = Column(Text, nullable=True)
    suggested_questions = Column(Text, nullable=True)
    ai_followup = Column(Text, nullable=True)
    feedback = Column(String, nullable=True)

    created_at = Column(TIMESTAMP, server_default=func.now())

    thread = relationship("Thread", back_populates="turns")

class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(String, nullable=False, index=True)
    feedback = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

def init_db():
    
    _ensure_schema()

def get_db():
    _ensure_schema()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
