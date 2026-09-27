# POS: guía de reproducción

El experimento compara BERT-base-uncased congelado con ajuste parcial de las dos últimas capas. El clasificador es lineal en ambos casos. La decisión se toma por accuracy de palabras en validación; macro-F1 se informa para las 17 clases UPOS.

## Ejecución
Desde la raíz del proyecto, con el entorno usado en AG News/NER activo:

```powershell
python src/pos_experiment.py --smoke --output runs/pos_smoke_nuevo
python src/pos_experiment.py --output runs/pos_nuevo
python -m unittest discover -s tests -p test_pos.py
```

El directorio de salida debe ser nuevo. Para reproducir los mismos modelos y datos, pasar `--config runs/pos_verified/config.json`. El SHA de BERT y la etiqueta r2.15 de UD se guardan en configuración; los CoNLL-U originales y sus SHA256 quedan en cada ejecución. Las dependencias exactas están en `runs/pos_verified/requirements-lock.txt` (PyTorch CUDA usa índice cu124). Se reutilizó el entorno de AG News sin instalar paquetes adicionales.

Tres épocas, semilla 42, microbatch 8 y acumulación 2 (efectivo 16), cabeza LR 1e-3, encoder entrenable 2e-5. Atención eager; bf16 si la GPU lo soporta. La pérdida se pondera por palabras en toda la acumulación. Se comprueba por hash que los parámetros congelados no cambien.

## Datos y límites
UD English EWT r2.15, particiones oficiales. CoNLL-U contiene filas multipalabra y nodos vacíos: se excluyen estos registros, conservando todas las palabras sintácticas con IDs enteros. Se aborta ante UPOS ausente/desconocido. Solo el primer subtoken tiene etiqueta, otras posiciones usan -100. Padding dinámico; las frases que excedan la capacidad provocarían un error explícito, nunca truncamiento silencioso. En esta ejecución ninguna la excede.

La primera época gana los empates entre épocas; entre métodos gana el de menos parámetros entrenables. Test solo se evalúa una vez por checkpoint elegido tras guardar selection.json. No hay búsqueda de hiperparámetros con test. Una semilla no demuestra significación estadística.

## Evidencia
`runs/pos_verified`: config, environment, source_files, data_manifest, alignment_check, historiales por método, predicciones JSONL por palabra, métricas por clase, análisis de errores, curvas y exportación estándar `delivered_model_pos`.

El notebook ejecutado lee estos artefactos, recalcula métricas independientemente y carga el modelo para una nueva inferencia CPU. No representa una segunda ejecución de entrenamiento. Para inferencia equivalente usa `predict_words` del script y palabras pretokenizadas; no promedies subtokens.

Modelo inglés, no validado en español. Publicado en Hugging Face; ver publication_manifest.json. Fuentes: https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15 y https://universaldependencies.org/format.html.
