"""
demo_server.py
---------------
Runs the REAL FastAPI app (main.app, unmodified) but swaps the MongoDB
dependency for an in-memory mongomock instance, so the whole API is
click-through-able at /docs without installing MongoDB first.

Data lives only in this process's memory and resets when the server
stops - this is for demos/grading, not for actual use. For real
persistence, just run `uvicorn main:app --reload` once MongoDB
Community Server is installed and running locally (see README).
"""

import mongomock
import uvicorn

from database.mongodb import MLReadyDatabase
from dependencies import get_db
from main import app

_demo_db = MLReadyDatabase(client=mongomock.MongoClient())
app.dependency_overrides[get_db] = lambda: _demo_db

if __name__ == "__main__":
    print("Starting MLReady API in DEMO mode (in-memory DB, no MongoDB needed)")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
