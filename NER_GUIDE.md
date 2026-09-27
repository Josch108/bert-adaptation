# NER: ajuste parcial frente a completo

## Entorno y ejecución

Usar el entorno Python 3.12 de AG News. Instalar `pip install -r requirements-ner.txt`.
En este equipo se encuentra en `C:\Users\IGNITER\Documents\ChatGPT\Cosas Escuela 2\.venv-agnews`.
Para un entorno nuevo con GPU, seguir primero las instrucciones CUDA en `AGNEWS_GUIDE.md`.

Desde la raíz del proyecto:

```powershell
python tests/test_ner.py
python src/ner_experiment.py --smoke --output runs/ner_smoke_01
python src/ner_experiment.py --output runs/ner_01
python src/ner_experiment.py --config runs/ner_01/config.json --output runs/ner_replicated
```

También puede ejecutarse `notebooks/02_ner_conll2003.ipynb` con el kernel del entorno.
En Colab subir/clonar el repositorio completo (incluidos ambos módulos en `src`) e instalar las dependencias.
Cada carpeta de salida debe ser nueva: no se sobrescriben experimentos anteriores.

## Protocolo

- `google-bert/bert-base-cased` y `lhoestq/conll2003` en Parquet, sin scripts remotos.
- Se fijan y registran commits del modelo y dataset en `config.json` y `data_manifest.json`.
- Se conservan las tres particiones oficiales y se auditan frases repetidas entre ellas. No se garantiza ausencia de textos idénticos.
- Todas las palabras se evalúan: sin truncamiento. Si una secuencia supera la capacidad de BERT, el experimento falla explícitamente.
- Primera subpalabra supervisada; resto, tokens especiales y padding: `-100`.
- Se imprime y guarda un batch de alineación real en `alignment_check.csv`.
- Ajuste parcial: últimas dos capas + cabeza; embeddings y primeras diez capas congeladas, con dropout desactivado en la parte congelada.
- Ajuste completo: todas las capas + cabeza. Misma inicialización de cabeza y orden de lotes, semilla 42.
- Tres épocas, microbatch 8 y acumulación 2; padding dinámico. Gradientes y pérdida ponderados por palabras supervisadas.
- LR cabeza 1e-3 y encoder entrenable 2e-5; warmup 10%, decaimiento lineal, AdamW, clipping 1.
- Criterio predefinido: mayor F1 micro de entidades de validación con `seqeval` strict IOB2. Empate exacto: menos parámetros entrenables.
- La selección se guarda antes de evaluar los mejores checkpoints en test. Se conserva también F1 estilo conlleval, como métrica secundaria explícitamente distinta.
- Una semilla por configuración: no se afirma significancia estadística. Los resultados no demuestran que un método siempre sea mejor.

## Artefactos y uso

La ejecución completa produce `delivered_model_ner/`, las predicciones por palabra, métricas por tipo,
historiales, curvas, configuración, versiones y análisis de errores.
Las predicciones se obtienen por argmax en la primera subpalabra, sin corregir automáticamente BIO.
Para inferencia equivalente, utilizar `predict_words(model, tokenizer, words)` de `src/ner_experiment.py`.
Un pipeline genérico con otra agregación de subtokens puede producir resultados diferentes.

Los modelos se publicaron al cerrar la entrega. `report/report.pdf` es el reporte final en inglés;
consulta también `report/NER_RESULTS.md` para el análisis específico de NER.

Fuentes: https://huggingface.co/datasets/lhoestq/conll2003 ; https://github.com/chakki-works/seqeval ; https://arxiv.org/abs/1810.04805
