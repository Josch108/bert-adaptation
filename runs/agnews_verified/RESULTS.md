# AG News: resultados medidos

Comparacion final de dos configuraciones; una semilla por configuracion.

Equipo: NVIDIA GeForce RTX 4060 Ti; precision: bf16.
Datos: 12000 entrenamiento / 2000 validacion / 2000 prueba; semilla 42.
Misma arquitectura BERT + pooler + clasificador lineal para aislar el efecto de congelar o ajustar el cuerpo.
Los tiempos incluyen el forward de BERT en cada epoca; no hay cache de embeddings.

| Metodo | Epoca elegida | F1 validacion | Accuracy prueba | F1 prueba | Entrenamiento (s) |
|---|---:|---:|---:|---:|---:|
| feature_based | 3 | 0.7733 | 0.7690 | 0.7683 | 91.8 |
| full_finetuning | 3 | 0.9334 | 0.9250 | 0.9251 | 309.0 |

Seleccion por validacion: **full_finetuning**. Diferencia de F1: 0.1601.
Una semilla no permite afirmar superioridad estadistica; diferencias pequenas pueden depender del azar.
Se evaluaron los mejores checkpoints de ambas alternativas en prueba despues de fijar la seleccion.
No se afirma una causa linguistica de las diferencias sin analizar ejemplos y realizar experimentos adicionales.

Evidencia: config.json, environment.json, splits.json, selection.json, comparison.csv, historiales por epoca y test_predictions.jsonl.
Los resultados de NER, POS y QA del reporte anterior siguen pendientes de verificacion.
