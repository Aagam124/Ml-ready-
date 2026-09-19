"""
MLReady - shared configuration constants.
Phase 1: Data Science engine thresholds.
Phase 2: MongoDB connection settings.
"""

import os

RANDOM_STATE = 42

# --- MongoDB (Phase 2) ---------------------------------------------------
# Override via environment variables when deploying; defaults target a local
# MongoDB Community Server instance with no auth, as used in dev/testing.
MONGO_URI = os.environ.get("MLREADY_MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MLREADY_DB_NAME", "mlready")

# PBKDF2 password hashing (Phase 2 users collection) - stdlib only, no extra
# native-build dependency like bcrypt requires.
PBKDF2_ITERATIONS = 260_000
PBKDF2_SALT_BYTES = 16

# --- JWT auth (Phase 3) ---------------------------------------------------
# In production, set MLREADY_JWT_SECRET to a long random value - never
# commit a real secret. This default is only for local dev.
JWT_SECRET_KEY = os.environ.get("MLREADY_JWT_SECRET", "dev-only-secret-change-me-before-deploying-32b")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24  # 24 hours

# --- File uploads (Phase 3) ------------------------------------------------
UPLOAD_DIR = os.environ.get("MLREADY_UPLOAD_DIR", "uploads")
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_UPLOAD_SIZE_MB = 50

# Data quality thresholds
NEAR_CONSTANT_THRESHOLD = 0.98      # one value makes up >=98% of non-null rows
HIGH_CARDINALITY_RATIO = 0.9        # unique values / row count
HIGH_CARDINALITY_MIN_UNIQUE = 50    # also require an absolute floor so small datasets
                                     # don't get flagged just for having few rows

# Outlier detection thresholds
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
ISOLATION_FOREST_CONTAMINATION = 0.05

# ML risk thresholds
CLASS_IMBALANCE_RATIO = 4.0         # majority:minority ratio above this = flagged
MULTICOLLINEARITY_CORR_THRESHOLD = 0.90
VIF_THRESHOLD = 10.0
LEAKAGE_CORR_THRESHOLD = 0.95
# Substring keywords: safe to match anywhere in the column name (unlikely
# to false-positive on ordinary column names).
LEAKAGE_NAME_SUBSTRINGS = ["target", "label", "outcome", "result", "leak", "future"]
# Whole-token keywords: only flagged when they appear as an entire
# underscore/space/dash-separated token, e.g. "y", "post", "after", "class" -
# this avoids matching things like "tenure_years", "postal_code", "afternoon".
LEAKAGE_NAME_TOKENS = ["y", "post", "after", "class"]
ID_LIKE_UNIQUE_RATIO = 0.98         # column where almost every value is unique

# Readiness score bands
SCORE_BANDS = [
    (80, 100, "Excellent"),
    (60, 79, "Good"),
    (40, 59, "Needs Improvement"),
    (0, 39, "Poor"),
]
