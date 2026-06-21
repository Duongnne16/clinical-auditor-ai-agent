from backend.app.core.config import get_settings


def get_doctor_id() -> str:
    """Development dependency to be replaced by JWT authentication."""
    return get_settings().fake_doctor_id
