"""Preprocesamiento de texto para tokenizacion y embeddings distribucionales."""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def ensure_nltk_resources(resources: Iterable[str] | None = None) -> None:
    """Verificar o descargar recursos livianos de NLTK cuando el usuario lo pida.

    La descarga debe ser explicita desde notebook o script, no automatica al
    importar el modulo.
    """

    raise NotImplementedError("Pendiente: preparar recursos de NLTK.")


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalizar texto antes de procesos posteriores.

    Puede incluir conversion a minusculas, limpieza de espacios y normalizacion
    unicode si el analisis lo requiere.
    """

    raise NotImplementedError("Pendiente: normalizar texto.")


def clean_tokens_for_gensim(
    tokens: Iterable[str],
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
    remove_punctuation: bool = True,
) -> list[str]:
    """Limpiar tokens para entrenamiento con Gensim.

    Debe eliminar puntuacion, numeros, stopwords y tokens vacios segun los
    parametros configurados.
    """

    raise NotImplementedError("Pendiente: limpiar tokens para Gensim.")


def sentence_to_gensim_tokens(sentence: str) -> list[str]:
    """Convertir una sentencia cruda en una lista de tokens apta para Gensim."""

    raise NotImplementedError("Pendiente: tokenizar sentencia para Gensim.")


def iter_spanish_billion_sentences(streaming: bool = True) -> Iterator[str]:
    """Iterar sentencias del corpus Spanish Billion Clean.

    Debe priorizar ``streaming=True`` para evitar cargar el corpus completo en
    memoria.
    """

    raise NotImplementedError("Pendiente: iterar Spanish Billion Clean.")
