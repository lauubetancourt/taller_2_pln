"""Preprocesamiento de texto para tokenizacion y embeddings distribucionales."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import re
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .config import (
    SPANISH_BILLION_DATASET_NAME,
    SPANISH_BILLION_SPLIT,
    SPANISH_BILLION_TEXT_COLUMN,
)


SPANISH_WORD_PATTERN = re.compile(r"[a-záéíóúüñ]+(?:-[a-záéíóúüñ]+)?", re.IGNORECASE)


def ensure_nltk_resources(resources: Iterable[str] | None = None) -> None:
    """Verificar o descargar recursos livianos de NLTK bajo demanda."""

    import nltk

    resource_names = list(resources or ["corpora/stopwords"])
    for resource in resource_names:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalizar texto conservando caracteres propios del espanol.

    La funcion convierte a minusculas por defecto y colapsa espacios, pero no
    elimina tildes ni transforma ``ñ``.
    """

    normalized = str(text)
    if lowercase:
        normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_spanish_stopwords() -> set[str]:
    """Cargar stopwords en espanol desde NLTK, descargandolas si faltan."""

    ensure_nltk_resources(["corpora/stopwords"])
    from nltk.corpus import stopwords

    return set(stopwords.words("spanish"))


def is_valid_token(token: str, stopwords: set[str] | None = None) -> bool:
    """Determinar si un token limpio es util para entrenar embeddings."""

    candidate = token.strip().lower()
    if not candidate:
        return False
    if stopwords and candidate in stopwords:
        return False
    if candidate.isnumeric() or any(character.isdigit() for character in candidate):
        return False
    if not re.search(r"[a-záéíóúüñ]", candidate, re.IGNORECASE):
        return False
    if re.fullmatch(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]+", candidate):
        return False
    if len(candidate) < 2:
        return False
    return True


def tokenize_clean_sentence(
    text: str,
    stopwords: set[str] | None = None,
) -> list[str]:
    """Convertir una sentencia cruda en tokens aptos para Gensim."""

    normalized = normalize_text(text)
    tokens = SPANISH_WORD_PATTERN.findall(normalized)
    return [token for token in tokens if is_valid_token(token, stopwords=stopwords)]


def clean_tokens_for_gensim(
    tokens: Iterable[str],
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
    remove_punctuation: bool = True,
) -> list[str]:
    """Limpiar una secuencia de tokens ya segmentados para Gensim."""

    stopwords = load_spanish_stopwords() if remove_stopwords else None
    cleaned_tokens: list[str] = []
    for token in tokens:
        candidate = normalize_text(token)
        if remove_numbers and any(character.isdigit() for character in candidate):
            continue
        if remove_punctuation and not re.search(r"[a-záéíóúüñ]", candidate):
            continue
        if is_valid_token(candidate, stopwords=stopwords):
            cleaned_tokens.append(candidate)
    return cleaned_tokens


def sentence_to_gensim_tokens(sentence: str) -> list[str]:
    """Convertir una sentencia cruda en una lista de tokens apta para Gensim."""

    return tokenize_clean_sentence(sentence, stopwords=load_spanish_stopwords())


def iter_streaming_dataset(
    dataset_name: str = SPANISH_BILLION_DATASET_NAME,
    split: str = SPANISH_BILLION_SPLIT,
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    """Cargar un dataset de Hugging Face como iterable en streaming."""

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    yield from dataset


def iter_spanish_billion_sentences(streaming: bool = True) -> Iterator[str]:
    """Iterar textos crudos de Spanish Billion Clean sin cargar todo en memoria."""

    dataset = iter_streaming_dataset(
        SPANISH_BILLION_DATASET_NAME,
        SPANISH_BILLION_SPLIT,
        streaming=streaming,
    )
    for row in dataset:
        text = row.get(SPANISH_BILLION_TEXT_COLUMN)
        if text:
            yield str(text)


def iter_preprocessed_sentences(
    dataset_iterable: Iterable[dict[str, Any]],
    text_column: str = SPANISH_BILLION_TEXT_COLUMN,
    stopwords: set[str] | None = None,
    max_sentences: int | None = None,
    min_tokens: int = 2,
) -> Iterator[list[str]]:
    """Iterar sentencias limpias desde un dataset en streaming."""

    progress = tqdm(total=max_sentences, desc="Preprocesando", unit="sent") if max_sentences else None
    yielded = 0
    try:
        for row in dataset_iterable:
            if max_sentences is not None and yielded >= max_sentences:
                break

            text = row.get(text_column)
            if not text:
                continue

            tokens = tokenize_clean_sentence(str(text), stopwords=stopwords)
            if len(tokens) < min_tokens:
                continue

            yielded += 1
            if progress:
                progress.update(1)
            yield tokens
    finally:
        if progress:
            progress.close()


def save_sentences_jsonl(sentences: Iterable[list[str]], output_path: str | Path) -> int:
    """Guardar sentencias tokenizadas en JSONL y retornar cuantas se guardaron."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for tokens in sentences:
            file.write(json.dumps({"tokens": tokens}, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_sentences_jsonl(input_path: str | Path) -> list[list[str]]:
    """Cargar sentencias limpias desde un archivo JSONL."""

    path = Path(input_path)
    sentences: list[list[str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = row.get("tokens", [])
            if tokens:
                sentences.append([str(token) for token in tokens])
    return sentences


def compute_preprocessing_stats(sentences: Iterable[list[str]]) -> dict[str, Any]:
    """Calcular estadisticas descriptivas del corpus preprocesado."""

    sentence_lengths: list[int] = []
    token_counter: Counter[str] = Counter()
    for sentence in sentences:
        sentence_lengths.append(len(sentence))
        token_counter.update(sentence)

    n_sentences = len(sentence_lengths)
    total_tokens = sum(sentence_lengths)
    stats = {
        "n_sentencias": n_sentences,
        "total_tokens": total_tokens,
        "longitud_promedio": round(total_tokens / n_sentences, 3) if n_sentences else 0,
        "longitud_minima": min(sentence_lengths) if sentence_lengths else 0,
        "longitud_maxima": max(sentence_lengths) if sentence_lengths else 0,
        "vocabulario_unico": len(token_counter),
        "top_30_tokens": token_counter.most_common(30),
    }
    return stats


def preprocessing_stats_to_dataframe(stats: dict[str, Any]) -> pd.DataFrame:
    """Convertir estadisticas de preprocesamiento a tabla plana."""

    rows = [
        {"metrica": key, "valor": value}
        for key, value in stats.items()
        if key != "top_30_tokens"
    ]
    rows.extend(
        {
            "metrica": f"top_token_{rank}",
            "valor": f"{token}:{count}",
        }
        for rank, (token, count) in enumerate(stats.get("top_30_tokens", []), start=1)
    )
    return pd.DataFrame(rows)


def collect_raw_and_clean_examples(
    dataset_iterable: Iterable[dict[str, Any]],
    text_column: str,
    stopwords: set[str] | None,
    n_examples: int = 10,
) -> pd.DataFrame:
    """Recolectar ejemplos texto original vs tokens limpios."""

    rows: list[dict[str, Any]] = []
    for row in dataset_iterable:
        text = row.get(text_column)
        if not text:
            continue
        raw_text = str(text)
        clean_tokens = tokenize_clean_sentence(raw_text, stopwords=stopwords)
        rows.append(
            {
                "texto_original": raw_text,
                "tokens_limpios": clean_tokens,
                "longitud_original_aprox": len(raw_text.split()),
                "longitud_limpia": len(clean_tokens),
            }
        )
        if len(rows) >= n_examples:
            break
    return pd.DataFrame(rows)


def preprocess_sentences_parallel(
    texts: list[str],
    stopwords: set[str] | None = None,
    n_workers: int = 2,
) -> list[list[str]]:
    """Preprocesar listas materializadas pequenas o medianas en paralelo.

    No se recomienda usar esta funcion directamente sobre streaming completo,
    porque materializar todo el corpus puede saturar la memoria.
    """

    stopword_set = set(stopwords or [])
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(_tokenize_with_stopwords, [(text, stopword_set) for text in texts]))


def _tokenize_with_stopwords(args: tuple[str, set[str]]) -> list[str]:
    text, stopwords = args
    return tokenize_clean_sentence(text, stopwords=stopwords)
