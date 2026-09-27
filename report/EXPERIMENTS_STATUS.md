# Experimentos verificados: índice de resultados

Las cuatro tareas tienen comparaciones ejecutadas y modelos seleccionados por validación. Este índice se genera desde los archivos de resultados; no reutiliza cifras del borrador anterior.

| Tarea | Método | Elegido | Métrica principal en prueba | Complementaria | Parámetros entrenables | Entrenamiento (s) |
|---|---|:---:|---:|---:|---:|---:|
| AG News | feature_based |  | Accuracy: 76.90% | Macro-F1: 76.83% | 3,076 | 91.8 |
| AG News | full_finetuning | Sí | Accuracy: 92.50% | Macro-F1: 92.51% | 109,485,316 | 309.0 |
| NER | partial_finetuning |  | F1 estricto IOB2: 89.11% | Recall: 89.13% | 14,182,665 | 145.8 |
| NER | full_finetuning | Sí | F1 estricto IOB2: 91.50% | Recall: 91.87% | 107,726,601 | 441.9 |
| POS | frozen |  | Accuracy por palabra: 93.23% | Macro-F1: 86.51% | 13,073 | 79.9 |
| POS | partial_finetuning | Sí | Accuracy por palabra: 95.73% | Macro-F1: 89.75% | 14,188,817 | 110.3 |
| QA | partial_finetuning |  | F1 de respuesta: 61.75% | Exact Match: 47.80% | 14,177,282 | 305.7 |
| QA | full_finetuning | Sí | F1 de respuesta: 83.28% | Exact Match: 73.60% | 108,893,186 | 846.0 |

## Cómo leer la tabla

Las métricas se muestran en escala 0–100, pero miden cosas distintas. No se deben ordenar tareas por esas cifras: clasificar una noticia, detectar entidades, etiquetar palabras y extraer respuestas no son el mismo problema.

Los tiempos corresponden a entrenamiento y excluyen validación, descargas y guardado; no son latencia de inferencia. Los experimentos se ejecutaron en la misma GPU, sin entrenamientos simultáneos que compitieran por recursos. Las configuraciones de atención y longitudes pueden variar entre tareas: comparar costos principalmente dentro de cada tarea.

Se usó una semilla por configuración. Diferencias pequeñas pueden depender de la aleatoriedad; no se afirma significancia estadística. Las particiones, duplicados auditados, reglas de evaluación y limitaciones están documentados en cada reporte.

## Qué aportan POS y QA

POS asigna una categoría gramatical a cada palabra: sustantivo, verbo, adjetivo, etc. La comparación pregunta cuánto mejora esa decisión al permitir que las últimas dos capas de BERT aprendan de la tarea, frente a entrenar únicamente el clasificador.

QA recibe una pregunta y un texto, y busca dónde empieza y termina la respuesta en ese texto. La comparación pregunta si adaptar todas las capas mejora la extracción frente a adaptar las últimas dos. No genera respuestas nuevas; tampoco se evaluó su capacidad de rechazar preguntas sin respuesta.

Las predicciones guardadas de POS y QA se recalcularon de forma independiente y se verificó que el modelo exportado corresponda al checkpoint seleccionado. Evidencia: [POS_QA_VERIFICATION.json](POS_QA_VERIFICATION.json).

Para repetir esta comprobación local, desde la raíz del proyecto: `python src/verify_pos_qa_artifacts.py .`. No vuelve a entrenar ni necesita descargar datos; requiere los artefactos locales completos.

## Evidencia y guías

- AG News: [resultados y errores](AGNEWS_RESULTS.md), [guía](../AGNEWS_GUIDE.md).
- NER: [resultados y errores](NER_RESULTS.md), [guía](../NER_GUIDE.md).
- POS: [resultados y errores](POS_RESULTS.md), [guía](../POS_GUIDE.md).
- QA: [resultados y errores](QA_RESULTS.md), [guía](../QA_GUIDE.md).

## Estado de entrega

Completado: [reporte final en inglés](report.pdf), código, notebooks, evidencia y cuatro modelos/tokenizers públicos en Hugging Face. Consulta [DELIVERY.md](../DELIVERY.md) y [publication_manifest.json](../publication_manifest.json). Los borradores previos están archivados y no forman parte de la entrega.
