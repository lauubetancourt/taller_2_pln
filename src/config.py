"""Configuracion central del proyecto.

Este modulo solo define rutas y constantes. No debe descargar datos, cargar modelos
ni ejecutar procesamiento pesado al importarse.
"""

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PDFS_DIR = DATA_DIR / "pdfs"
CONLL2002_RAW_DIR = RAW_DATA_DIR / "conll2002"
ANCORA_RAW_DIR = RAW_DATA_DIR / "ancora"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

SEED = 42
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)
BATCH_SIZE_TOKENIZATION = 1000
BATCH_SIZE_SENTENCE_EMBEDDINGS = 32
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
