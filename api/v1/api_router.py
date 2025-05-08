from fastapi import APIRouter
from api.v1.endpoints import  process, result

api_router = APIRouter()

api_router.include_router(process.router, prefix="/process", tags=["Traitement"])
api_router.include_router(result.router, prefix="/result", tags=["Résultat"])
