from backend.app.db.models import PrescriptionHistory, ReportHistory, User


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


def test_audit_history_models_declare_expected_tables() -> None:
    prescription_columns = PrescriptionHistory.__table__.columns
    report_columns = ReportHistory.__table__.columns

    assert PrescriptionHistory.__tablename__ == "prescription_history"
    assert ReportHistory.__tablename__ == "report_history"
    for name in (
        "id",
        "doctor_id",
        "prescription_text",
        "patient_context",
        "query_types",
        "use_gemini",
        "top_k_per_type",
        "status",
        "overall_risk_level",
        "warnings",
        "errors",
        "audit_payload",
        "created_at",
    ):
        assert name in prescription_columns
    for name in (
        "id",
        "doctor_id",
        "prescription_history_id",
        "report_status",
        "summary",
        "doctor_facing_response",
        "report_payload",
        "created_at",
    ):
        assert name in report_columns
    assert prescription_columns["doctor_id"].index is True
    assert prescription_columns["created_at"].index is True
    assert report_columns["doctor_id"].index is True
    assert report_columns["prescription_history_id"].index is True
    assert report_columns["created_at"].index is True
