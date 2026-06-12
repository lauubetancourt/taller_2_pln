"""Entrenar Word2Vec y FastText sobre sentencias limpias JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    WORD_EMBEDDING_MIN_COUNT,
    WORD_EMBEDDING_VECTOR_SIZE,
    WORD_EMBEDDING_WINDOW,
    WORD_EMBEDDINGS_OUTPUT_DIR,
    WORD_EMBEDDINGS_REPORTS_DIR,
)
from src.preprocessing import compute_preprocessing_stats, load_sentences_jsonl
from src.word_embeddings import (
    get_vocabulary_stats,
    save_embedding_matrix,
    save_model,
    save_vocab_stats_csv,
    train_fasttext,
    train_word2vec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences-file", required=True)
    parser.add_argument("--vector-size", type=int, default=WORD_EMBEDDING_VECTOR_SIZE)
    parser.add_argument("--window", type=int, default=WORD_EMBEDDING_WINDOW)
    parser.add_argument("--min-count", type=int, default=WORD_EMBEDDING_MIN_COUNT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--save-embedding-matrix", action="store_true")
    parser.add_argument("--max-words-matrix", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences_path = Path(args.sentences_file)
    output_suffix = args.output_suffix or _infer_suffix(sentences_path)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WORD_EMBEDDINGS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    WORD_EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences_jsonl(sentences_path)
    if not sentences:
        raise ValueError(f"No se cargaron sentencias desde {sentences_path}.")

    stats = compute_preprocessing_stats(sentences)
    print("=== Corpus limpio ===")
    print(f"Archivo: {sentences_path}")
    print(f"Sentencias: {stats['n_sentencias']}")
    print(f"Tokens: {stats['total_tokens']}")
    print(f"Vocabulario unico: {stats['vocabulario_unico']}")
    print(f"workers: {args.workers}")

    print("\n=== Entrenando Word2Vec ===")
    word2vec_model = train_word2vec(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
    )
    word2vec_path = MODELS_DIR / f"word2vec_{output_suffix}.model"
    save_model(word2vec_model, word2vec_path)
    print(f"Word2Vec guardado: {word2vec_path}")

    print("\n=== Entrenando FastText ===")
    fasttext_model = train_fasttext(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
    )
    fasttext_path = MODELS_DIR / f"fasttext_{output_suffix}.model"
    save_model(fasttext_model, fasttext_path)
    print(f"FastText guardado: {fasttext_path}")

    stats_path = WORD_EMBEDDINGS_REPORTS_DIR / f"model_vocab_stats_{output_suffix}.csv"
    save_vocab_stats_csv(
        [get_vocabulary_stats(word2vec_model), get_vocabulary_stats(fasttext_model)],
        stats_path,
    )
    print(f"Estadisticas de vocabulario: {stats_path}")

    if args.save_embedding_matrix:
        save_embedding_matrix(
            word2vec_model,
            WORD_EMBEDDINGS_OUTPUT_DIR / f"word2vec_matrix_{output_suffix}",
            max_words=args.max_words_matrix,
        )
        save_embedding_matrix(
            fasttext_model,
            WORD_EMBEDDINGS_OUTPUT_DIR / f"fasttext_matrix_{output_suffix}",
            max_words=args.max_words_matrix,
        )
        print(f"Matrices guardadas en: {WORD_EMBEDDINGS_OUTPUT_DIR}")


def _infer_suffix(sentences_path: Path) -> str:
    stem = sentences_path.stem
    return stem.removeprefix("spanish_billion_clean_")


if __name__ == "__main__":
    main()
