# Python AI Service (Milestone 2)

Development stub providing `/health` and `POST /analyze` with verified video file metadata, **not fighter recognition**.

Windows PowerShell from the `ai-service` directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:VIDEO_STORAGE_ROOT = (Resolve-Path ..\backend\storage\videos).Path
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The backend's `AI_SERVICE_URL` defaults to `http://localhost:8000`. Keep this service local; it is not authenticated. The `storagePath` must be inside `VIDEO_STORAGE_ROOT`. For multi-host deployment replace shared local path access with object storage.
