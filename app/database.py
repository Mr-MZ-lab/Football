from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

_url = settings.DATABASE_URL
_is_sqlite = _url.startswith("sqlite")

engine = create_engine(
    _url,
    pool_pre_ping=not _is_sqlite,
    # SQLite uses StaticPool — pool_size/max_overflow not applicable
    **({} if _is_sqlite else {"pool_size": 10, "max_overflow": 20}),
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for FastAPI routes — yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables():
    """Create all tables defined in models. Called on app startup."""
    from app.models import team, player, match, prediction  # noqa: F401
    Base.metadata.create_all(bind=engine)
