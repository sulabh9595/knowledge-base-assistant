from fastapi import APIRouter
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse)
def root():
    return {"message": "Knowledge Base Assistant API is running"}


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"message": "ok"}
