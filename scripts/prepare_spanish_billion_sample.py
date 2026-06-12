"""Preparar muestras limpias del corpus Spanish Billion Clean."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_SENTENCES_DIR,
    SPANISH_BILLION_DATASET_NAME,
    SPANISH_BILLION_SPLIT,
    SPANISH_BILLION_TEXT_COLUMN,
    WORD_EMBEDDINGS_REPORTS_DIR,
)
from src.preprocessing import (
    collect_raw_and_clean_examples,
    compute_preprocessing_stats,
    iter_preprocessed_sentences,
    iter_streaming_dataset,
    load_spanish_stopwords,
    preprocessing_stats_to_dataframe,
    save_sentences_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=SPANISH_BILLION_DATASET_NAME)
    parser.add_argument("--split", default=SPANISH_BILLION_SPLIT)
    parser.add_argument("--text-column", default=SPANISH_BILLION_TEXT_COLUMN)
    parser.add_argument("--max-sentences", type=int, default=10000)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--min-tokens", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or f"sample_{args.max_sentences}"

    PROCESSED_SENTENCES_DIR.mkdir(parents=True, exist_ok=True)
    WORD_EMBEDDINGS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Preparacion Spanish Billion Clean ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Columna de texto: {args.text_column}")
    print(f"Max sentencias limpias: {args.max_sentences}")

    stopwords = load_spanish_stopwords()

    examples_iterable = iter_streaming_dataset(args.dataset_name, args.split, streaming=True)
    examples_df = collect_raw_and_clean_examples(
        examples_iterable,
        text_column=args.text_column,
        stopwords=stopwords,
        n_examples=10,
    )
    examples_path = (
        WORD_EMBEDDINGS_REPORTS_DIR
        / f"preprocessing_examples_{output_name}.csv"
    )
    examples_df.to_csv(examples_path, index=False)

    dataset_iterable = iter_streaming_dataset(args.dataset_name, args.split, streaming=True)
    sentences = list(
        iter_preprocessed_sentences(
            dataset_iterable,
            text_column=args.text_column,
            stopwords=stopwords,
            max_sentences=args.max_sentences,
            min_tokens=args.min_tokens,
        )
    )

    sentences_path = (
        PROCESSED_SENTENCES_DIR
        / f"spanish_billion_clean_{output_name}.jsonl"
    )
    save_sentences_jsonl(sentences, sentences_path)

    stats = compute_preprocessing_stats(sentences)
    stats_path = WORD_EMBEDDINGS_REPORTS_DIR / f"preprocessing_stats_{output_name}.csv"
    preprocessing_stats_to_dataframe(stats).to_csv(stats_path, index=False)

    print("\n=== Resumen ===")
    print(f"Sentencias procesadas: {stats['n_sentencias']}")
    print(f"Total de tokens: {stats['total_tokens']}")
    print(f"Vocabulario unico: {stats['vocabulario_unico']}")
    print(f"Ejemplos raw/clean: {examples_path}")
    print(f"Sentencias JSONL: {sentences_path}")
    print(f"Estadisticas: {stats_path}")


if __name__ == "__main__":
    main()
