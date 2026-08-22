from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.deps import get_db
from src.storage.database import DBManager

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: DBManager = Depends(get_db)):
    db_ok = db.ping()
    return JSONResponse(
        {"status": "ok" if db_ok else "degraded", "services": {"database": db_ok}},
        status_code=200 if db_ok else 503,
    )
