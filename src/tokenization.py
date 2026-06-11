"""Tokenizacion subword con WordPiece-BETO y SentencePiece-T5."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import BATCH_SIZE_TOKENIZATION


BETO_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
T5_SMALL_MODEL_NAME = "google-t5/t5-small"


def load_wordpiece_tokenizer(model_name: str = BETO_MODEL_NAME) -> Any:
    """Cargar el tokenizer WordPiece de BETO."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def load_sentencepiece_tokenizer(model_name: str = T5_SMALL_MODEL_NAME) -> Any:
    """Cargar el tokenizer SentencePiece asociado a T5."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def reconstruct_text_from_tokens(tokens: list[str]) -> str:
    """Reconstruir texto legible desde tokens originales separados por palabras."""

    text = " ".join(tokens)
    replacements = {
        " ,": ",",
        " .": ".",
        " ;": ";",
        " :": ":",
        " !": "!",
        " ?": "?",
        " )": ")",
        "( ": "(",
        " ]": "]",
        "[ ": "[",
        " }": "}",
        "{ ": "{",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def tokenize_sentences(
    sentences: list[list[str]] | list[str],
    tokenizer: Any,
    batch_size: int | None = BATCH_SIZE_TOKENIZATION,
) -> list[dict[str, Any]]:
    """Tokenizar sentencias por lotes con un tokenizer de Transformers."""

    texts = [
        reconstruct_text_from_tokens(sentence) if isinstance(sentence, list) else sentence
        for sentence in sentences
    ]
    outputs: list[dict[str, Any]] = []
    effective_batch_size = batch_size or len(texts) or 1

    for start in range(0, len(texts), effective_batch_size):
        batch_texts = texts[start : start + effective_batch_size]
        encoded_batch = tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        for text, input_ids, attention_mask in zip(
            batch_texts,
            encoded_batch["input_ids"],
            encoded_batch["attention_mask"],
            strict=True,
        ):
            outputs.append(
                {
                    "text": text,
                    "tokens": tokenizer.convert_ids_to_tokens(input_ids),
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

    return outputs


def tokenize_with_beto(
    sentences: list[list[str]] | list[str],
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    """Tokenizar sentencias con WordPiece-BETO."""

    beto_tokenizer = tokenizer or load_wordpiece_tokenizer()
    return tokenize_sentences(sentences, beto_tokenizer)


def tokenize_with_t5(
    sentences: list[list[str]] | list[str],
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    """Tokenizar sentencias con SentencePiece-T5."""

    t5_tokenizer = tokenizer or load_sentencepiece_tokenizer()
    return tokenize_sentences(sentences, t5_tokenizer)


def build_tokenization_comparison_dataframe(
    dataset_name: str,
    examples_by_split: dict[str, list[dict[str, Any]]],
    beto_tokenizer: Any,
    t5_tokenizer: Any,
) -> pd.DataFrame:
    """Construir una comparacion en DataFrame para ejemplos por split."""

    rows: list[dict[str, Any]] = []

    for split_name, examples in examples_by_split.items():
        original_tokens = [_extract_tokens(example) for example in examples]
        texts = [reconstruct_text_from_tokens(tokens) for tokens in original_tokens]
        beto_outputs = tokenize_with_beto(texts, tokenizer=beto_tokenizer)
        t5_outputs = tokenize_with_t5(texts, tokenizer=t5_tokenizer)

        for index, (example, text, beto_output, t5_output) in enumerate(
            zip(examples, texts, beto_outputs, t5_outputs, strict=True),
            start=1,
        ):
            tokens = _extract_tokens(example)
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "example_index": index,
                    "source_sentence_id": example.get("sentence_id"),
                    "source_file": example.get("source_file"),
                    "text": text,
                    "original_tokens": tokens,
                    "beto_model": BETO_MODEL_NAME,
                    "beto_tokens": beto_output["tokens"],
                    "beto_token_count": len(beto_output["tokens"]),
                    "t5_model": T5_SMALL_MODEL_NAME,
                    "t5_tokens": t5_output["tokens"],
                    "t5_token_count": len(t5_output["tokens"]),
                }
            )

    return pd.DataFrame(rows)


def compare_tokenized_examples(
    dataset_name: str,
    split_name: str,
    examples: list[dict[str, Any]],
    wordpiece_outputs: list[dict[str, Any]],
    sentencepiece_outputs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Construir una tabla comparativa para una particion ya tokenizada."""

    rows = []
    for index, (example, wordpiece, sentencepiece) in enumerate(
        zip(examples, wordpiece_outputs, sentencepiece_outputs, strict=True),
        start=1,
    ):
        tokens = _extract_tokens(example)
        rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "example_index": index,
                "source_sentence_id": example.get("sentence_id"),
                "source_file": example.get("source_file"),
                "text": reconstruct_text_from_tokens(tokens),
                "original_tokens": tokens,
                "beto_tokens": wordpiece["tokens"],
                "beto_token_count": len(wordpiece["tokens"]),
                "t5_tokens": sentencepiece["tokens"],
                "t5_token_count": len(sentencepiece["tokens"]),
            }
        )
    return pd.DataFrame(rows)


def _extract_tokens(example: dict[str, Any] | list[str] | str) -> list[str]:
    if isinstance(example, dict):
        tokens = example.get("tokens")
        if isinstance(tokens, list):
            return [str(token) for token in tokens]
        if "text" in example:
            return str(example["text"]).split()
    if isinstance(example, list):
        return [str(token) for token in example]
    return str(example).split()
