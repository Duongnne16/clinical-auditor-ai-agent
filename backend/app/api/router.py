from fastapi import APIRouter

from backend.app.api.routes import (
    auth,
    chat,
    clinical_workflow,
    doctor_notes,
    health,
    prescriptions,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(prescriptions.router)
api_router.include_router(chat.router)
api_router.include_router(clinical_workflow.router)
api_router.include_router(doctor_notes.router)
