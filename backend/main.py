"""
main.py
-------
MLReady backend entry point.

Run with:  uvicorn main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import analysis, auth, datasets, models, reports

app = FastAPI(
    title="MLReady API",
    description="Machine Learning Readiness & Risk Analyzer",
    version="0.6.0",  # Phase 6
)

# Dev-friendly CORS: the React dashboard (Phase 4) runs on a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(analysis.router)
app.include_router(models.router)
app.include_router(reports.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
