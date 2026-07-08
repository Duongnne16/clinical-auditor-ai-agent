from backend.app.db.models import User


def test_user_model_declares_auth_foundation_columns() -> None:
    columns = User.__table__.columns

    assert User.__tablename__ == "users"
    for name in (
        "id",
        "doctor_id",
        "email",
        "full_name",
        "hashed_password",
        "is_active",
        "created_at",
        "updated_at",
    ):
        assert name in columns
    assert columns["doctor_id"].unique is True
    assert columns["email"].unique is True
    assert columns["doctor_id"].index is True
    assert columns["email"].index is True
