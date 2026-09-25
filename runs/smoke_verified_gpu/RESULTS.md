# AG News: resultados medidos

PRUEBA TECNICA: no usar estas cifras como resultados finales.

Equipo: NVIDIA GeForce RTX 4060 Ti; precision: bf16.
Datos: 32 entrenamiento / 16 validacion / 16 prueba; semilla 42.
Misma arquitectura BERT + pooler + clasificador lineal para aislar el efecto de congelar o ajustar el cuerpo.
Los tiempos incluyen el forward de BERT en cada epoca; no hay cache de embeddings.

| Metodo | Epoca elegida | F1 validacion | Accuracy prueba | F1 prueba | Entrenamiento (s) |
|---|---:|---:|---:|---:|---:|
| feature_based | 1 | 0.1053 | 0.2500 | 0.1000 | 3.0 |
| full_finetuning | 1 | 0.1000 | 0.2500 | 0.1000 | 0.6 |

Seleccion por validacion: **feature_based**. Diferencia de F1: 0.0053.
Una semilla no permite afirmar superioridad estadistica; diferencias pequenas pueden depender del azar.
Se evaluaron los mejores checkpoints de ambas alternativas en prueba despues de fijar la seleccion.
No se afirma una causa linguistica de las diferencias sin analizar ejemplos y realizar experimentos adicionales.

Evidencia: config.json, environment.json, splits.json, selection.json, comparison.csv, historiales por epoca y test_predictions.jsonl.
Los resultados de NER, POS y QA del reporte anterior siguen pendientes de verificacion.
