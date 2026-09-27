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
Las cuatro tareas están completadas; consulta EXPERIMENTS_STATUS.md.

## Lectura de los resultados

La selección favoreció **full_finetuning** antes de evaluar prueba. En este experimento,
el ajuste completo acertó 1,850 de 2,000 noticias.
La mejora observada frente al extractor congelado es de
15.60 puntos porcentuales de accuracy,
con 3.37 veces el tiempo de entrenamiento.
Son resultados para esta arquitectura, partición y configuración; no demuestran que todo ajuste completo
supere a cualquier extractor congelado. No se optimizaron exhaustivamente ambas alternativas.

### Errores y límites

- De 150 errores del ganador, 77 son confusiones entre Business y Sci/Tech
  (51 Business -> Sci/Tech, 26 Sci/Tech -> Business).
- Sports tuvo 488 aciertos de 500; Business, 435 de 500.
- Algunos ejemplos mezclan temas empresariales y tecnológicos. Por ejemplo, la noticia sobre
  desconocimiento del spyware por consumidores tiene etiqueta Business y se predijo Sci/Tech.
  Esto ilustra una frontera temática ambigua; no demuestra que todas las etiquetas sean erróneas.
- La pérdida de validación del ajuste completo pasó de 0.2430 en época 2 a 0.2743 en época 3,
  aunque macro-F1 aumentó. Las métricas capturan aspectos distintos: cross-entropy también depende
  de las probabilidades asignadas. Se mantuvo el criterio predefinido de macro-F1, sin cambiarlo al ver prueba.
- Una sola semilla; no se estimaron intervalos de confianza ni se probaron otros dominios o idiomas.
- Los resultados describen muestras de AG News, no una garantía de rendimiento en una empresa.

## Reproducibilidad y comprobaciones

Se guardaron versiones exactas de paquetes, commits del modelo/dataset, índices originales y configuración.
Se excluyeron 161 filas del train oficial por duplicación de texto normalizado dentro del conjunto
o respecto al test oficial, antes del muestreo. Validación y prueba tienen 500 ejemplos por categoría.

Se comprobó que el cuerpo congelado conservó exactamente sus pesos, que los checkpoints se recargan,
que las métricas coinciden al recalcularlas desde las 2,000 predicciones guardadas y que el modelo
exportado es idéntico al checkpoint seleccionado. El notebook terminó sus seis celdas de código sin errores.

En inferencia local, los dos ejemplos de comprobación se clasificaron como Sports y Sci/Tech.
Es una prueba de funcionamiento, adicional a la evaluación cuantitativa.

## Archivos

- [Notebook ejecutado](../notebooks/01_topic_classification_agnews.ipynb)
- [Tabla de resultados](../runs/agnews_verified/comparison.csv)
- [Curvas](../runs/agnews_verified/learning_curves.png)
- [Configuración](../runs/agnews_verified/config.json)
- [Selección por validación](../runs/agnews_verified/selection.json)
- [Modelo y model card](../runs/agnews_verified/delivered_model_agnews/README.md)
- [Guía de ejecución](../AGNEWS_GUIDE.md)

El modelo está publicado; consulta publication_manifest.json en la raíz del proyecto.
El reporte final report.pdf incorpora estas mediciones; el borrador previo está archivado.
