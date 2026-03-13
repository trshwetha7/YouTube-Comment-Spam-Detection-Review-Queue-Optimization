"""Project-wide configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"

RANDOM_STATE = 42
TEST_SIZE = 0.25
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_REVIEW_THRESHOLD = 0.55
DEFAULT_AUTO_REMOVE_THRESHOLD = 0.85

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Multinomial Naive Bayes": "multinomial_nb.joblib",
    "Random Forest": "random_forest.joblib",
}

DATASET_VIDEO_MAP = {
    "Youtube01-Psy.csv": "PSY - GANGNAM STYLE",
    "Youtube02-KatyPerry.csv": "Katy Perry - Roar",
    "Youtube03-LMFAO.csv": "LMFAO - Party Rock Anthem",
    "Youtube04-Eminem.csv": "Eminem - Love The Way You Lie",
    "Youtube05-Shakira.csv": "Shakira - Waka Waka",
}
