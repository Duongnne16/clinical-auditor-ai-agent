# API contracts

All routes use the `/api/v1` prefix.

- `GET /health`: service status and version.
- `POST /prescriptions/audit`: accepts one or more drugs and returns a
  placeholder audit report.
- `POST /chat`: returns a disabled-integration message.
- `POST /doctor-notes`: stores a note for the development doctor in SQLite.
- `GET /doctor-notes`: lists notes for the development doctor.

The doctor identity currently comes from the `get_doctor_id` development
dependency. Responses that could be mistaken for clinical output include a
clear non-clinical disclaimer.
