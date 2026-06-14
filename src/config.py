"""Configuracion central del proyecto.

Este modulo solo define rutas y constantes. No debe descargar datos, cargar modelos
ni ejecutar procesamiento pesado al importarse.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PDFS_DIR = DATA_DIR / "pdfs"
CONLL2002_RAW_DIR = RAW_DATA_DIR / "conll2002"
ANCORA_RAW_DIR = RAW_DATA_DIR / "ancora"
PDF_CHUNKS_PATH = PROCESSED_DATA_DIR / "pdf_chunks.jsonl"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

SEED = 42
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)
BATCH_SIZE_TOKENIZATION = 1000
BATCH_SIZE_SENTENCE_EMBEDDINGS = 32
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SPANISH_BILLION_DATASET_NAME = "jhonparra18/spanish_billion_words_clean"
SPANISH_BILLION_SPLIT = "train"
SPANISH_BILLION_TEXT_COLUMN = "text"

MAX_SENTENCES_DEBUG = 1000
MAX_SENTENCES_SMALL = 10000
MAX_SENTENCES_MEDIUM = 50000

VISUALIZATION_SAMPLE_SIZES = [100, 5000, 10000]

PROCESSED_SENTENCES_DIR = PROCESSED_DATA_DIR / "sentences"
WORD_EMBEDDINGS_OUTPUT_DIR = OUTPUTS_DIR / "embeddings" / "word_embeddings"
WORD_EMBEDDINGS_REPORTS_DIR = OUTPUTS_DIR / "reports" / "word_embeddings"
WORD_EMBEDDINGS_FIGURES_DIR = FIGURES_DIR / "word_embeddings"

SENTENCE_EMBEDDINGS_OUTPUT_DIR = OUTPUTS_DIR / "embeddings" / "sentence_embeddings"
SENTENCE_EMBEDDINGS_REPORTS_DIR = OUTPUTS_DIR / "reports" / "sentence_embeddings"
SENTENCE_EMBEDDINGS_FIGURES_DIR = FIGURES_DIR / "sentence_embeddings"
SENTENCE_EMBEDDINGS_SIMILARITY_DIR = OUTPUTS_DIR / "similarity" / "sentence_embeddings"

WORD2VEC_MODEL_PATH = MODELS_DIR / "word2vec_spanish_billion.model"
FASTTEXT_MODEL_PATH = MODELS_DIR / "fasttext_spanish_billion.model"

WORD_EMBEDDING_VECTOR_SIZE = 300
WORD_EMBEDDING_WINDOW = 5
WORD_EMBEDDING_MIN_COUNT = 5
WORD_EMBEDDING_WORKERS = 4
