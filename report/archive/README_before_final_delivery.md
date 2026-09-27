# U2T01: Adapting BERT for NLP Tasks

Proyecto enfocado en la adaptación empírica de `bert-base` a través de los tres peldaños de la escalera de adaptación (*Feature-based*, *Partial fine-tuning*, *Full fine-tuning*) en cuatro tareas clásicas de NLP.

## Estado de los cuatro experimentos

AG News, NER, POS y QA están ejecutados y verificados. [Comparación medida de las cuatro tareas](report/EXPERIMENTS_STATUS.md). [POS](report/POS_RESULTS.md) y [QA](report/QA_RESULTS.md) incluyen métricas y análisis de errores.

Los cuatro modelos seleccionados están exportados localmente; ver `models_manifest.json`. Faltan el PDF integrado y la publicación en Hugging Face. **El reporte final debe redactarse en inglés.**

Los notebooks de POS y QA muestran evidencia ejecutada de los entrenamientos guardados. Para volver a entrenar, usa los comandos de [POS_GUIDE.md](POS_GUIDE.md) y [QA_GUIDE.md](QA_GUIDE.md), con una carpeta de salida nueva.

## Estado de la revisión de AG News

**AG News completado:** accuracy de prueba 76.90% (congelado) y 92.50% (ajuste completo). [Resultados verificados y análisis](report/AGNEWS_RESULTS.md).

AG News tiene un flujo reproducible con separación train/validation/test, selección por validación,
exportación del ganador y evidencia por ejecución. Consulta [AGNEWS_GUIDE.md](AGNEWS_GUIDE.md).
La implementación está en `src/agnews_experiment.py`; las dependencias específicas, en `requirements-agnews.txt`.

**Las cifras de `report/REPORT.md` y `report/report.pdf` son del borrador anterior y no están verificadas.**
Para AG News, usa `runs/agnews_verified/RESULTS.md` solo cuando su `status.json` indique `complete`.
POS y QA también están completados; consulta el índice de resultados verificados. Las salidas smoke son pruebas técnicas, no resultados finales.

## Estado de NER

**NER completado:** F1 estricto en prueba 0.8911 (parcial) y 0.9150 (completo). [Resultados verificados](report/NER_RESULTS.md) y [guía](NER_GUIDE.md). Modelo elegido por validación y exportado localmente; una semilla por configuración.

## Estructura del Proyecto

```
bert-adaptation/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_topic_classification_agnews.ipynb
│   ├── 02_ner_conll2003.ipynb
│   ├── 03_pos_tagging_ud_ewt.ipynb
│   └── 04_extractive_qa_squad.ipynb
├── report/
│   └── (borradores, gráficas y reporte PDF final)
└── src/
    ├── agnews_experiment.py
    ├── ner_experiment.py
    ├── pos_experiment.py
    ├── qa_experiment.py
    └── upload_models_to_hf.py
```

## Flujo de Trabajo en Google Colab

1. **Hardware:** En Google Colab, seleccionar entorno de ejecución con aceleración por GPU (según disponibilidad en Runtime -> Change runtime type).
2. **Reproducibilidad:** Cada notebook incluye semillas fijas (`seed=42`) y seguimiento de métricas (loss por época, tiempos en segundos y métricas de evaluación).
3. **Reglas de la Tarea:**
   - Comparación obligatoria de al menos 2 métodos por tarea (modelo entregado vs alternativa rechazada).
   - Cobertura de los tres métodos de adaptación a lo largo del proyecto.
   - Tasas de aprendizaje diferenciadas (`head ~ 1e-3`, `backbone ~ 2e-5`).
   - Alineación de subtokens con máscara `-100` y *sanity check* visual para tareas de secuencias.
   - Exportación de los 4 modelos ganadores a Hugging Face Hub con Model Cards completas.
