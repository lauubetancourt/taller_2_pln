# Taller PLN

Proyecto base para desarrollar localmente un taller de Procesamiento del Lenguaje Natural desde Jupyter Notebook, manteniendo la logica pesada en scripts y modulos reutilizables de Python.

El taller cubre tres bloques:

1. Tokenizacion de datasets con Conll2002, Ancora, WordPiece con BETO y SentencePiece con `t5-small`.
2. Entrenamiento y analisis de embeddings distribucionales con Word2Vec y FastText sobre Spanish Billion Clean en modo `streaming=True`.
3. Procesamiento de PDFs, chunking, embeddings semanticos y busqueda por similitud coseno.

## Estructura

```text
taller_pln/
├── notebooks/
│   └── taller_pln.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset_loading.py
│   ├── preprocessing.py
│   ├── tokenization.py
│   ├── word_embeddings.py
│   ├── pdf_processing.py
│   ├── sentence_embeddings.py
│   ├── similarity.py
│   └── visualization.py
├── scripts/
│   ├── run_tokenization.py
│   ├── train_word2vec.py
│   ├── train_fasttext.py
│   ├── process_pdfs.py
│   └── generate_sentence_embeddings.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── pdfs/
├── outputs/
│   ├── tokenization/
│   ├── embeddings/
│   ├── figures/
│   ├── similarity/
│   └── reports/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalacion

Desde la raiz del proyecto:

```bash
cd taller_pln
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name taller-pln --display-name "Python (taller-pln)"
```

En Windows PowerShell, la activacion del entorno cambia a:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Ejecutar el notebook

```bash
cd taller_pln
jupyter notebook notebooks/taller_pln.ipynb
```

El notebook debe funcionar como orquestador: probar funciones, mostrar ejemplos pequenos, generar tablas, graficar y redactar analisis. Las funciones costosas deben vivir en `src/` y ejecutarse desde scripts cuando sea necesario.

## Ejecutar scripts desde terminal

Ejemplos previstos para fases posteriores:

```bash
python scripts/run_tokenization.py
python scripts/train_word2vec.py
python scripts/train_fasttext.py
python scripts/process_pdfs.py
python scripts/generate_sentence_embeddings.py
```

En esta fase los scripts quedan como puntos de entrada base y no descargan datasets, no entrenan modelos y no generan embeddings.

## Nota sobre memoria

No cargues datasets grandes completos en memoria si no es necesario. Para corpus masivos como Spanish Billion Clean, usa `streaming=True`, procesamiento por lotes e iteradores. Ajusta `N_WORKERS`, tamanos de lote y subconjuntos de visualizacion segun la memoria y CPU de la maquina local.
