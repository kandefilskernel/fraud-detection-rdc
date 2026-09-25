from shared.database.session import make_session_factory

engine, SessionLocal = make_session_factory()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
