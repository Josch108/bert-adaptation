# POS: resultados medidos

Una semilla (42), tres épocas por configuración.
Particiones oficiales UD English EWT r2.15: {'train': 12544, 'validation': 2001, 'test': 2077}.
Máximos de subtokens: {'train': 176, 'validation': 90, 'test': 337}. Sin truncamiento.
Frases repetidas entre particiones, conservadas: {'train/validation': 36, 'train/test': 47, 'validation/test': 27}.
Selección por accuracy de palabras en validación; empate entre métodos: menos parámetros entrenables.
Prueba se evalúa después de guardar la selección. Macro-F1 incluye las 17 etiquetas UPOS.

| Método | Época | Accuracy validación | Accuracy prueba | Macro-F1 prueba | Entrenamiento s |
|---|---:|---:|---:|---:|---:|
| frozen | 3 | 0.9312 | 0.9323 | 0.8651 | 79.9 |
| partial_finetuning | 3 | 0.9588 | 0.9573 | 0.8975 | 110.3 |

Seleccionado: **partial_finetuning**.
Tiempos excluyen validación y guardado; no son latencia de inferencia.
La pérdida se pondera por palabras supervisadas. Solo el primer subtoken recibe etiqueta.
Las filas multipalabra y nodos vacíos de CoNLL-U no son palabras sintácticas y se omiten; etiquetas desconocidas abortan la carga.
Modelo base uncased: puede limitar distinciones apoyadas en mayúsculas como PROPN/NOUN.
Una semilla no demuestra superioridad estadística. Sin búsqueda de hiperparámetros usando prueba.

## Errores frozen
{"words": 25094, "errors": 1699, "correct": 23395}
```json
[
  {
    "gold": "PROPN",
    "prediction": "NOUN",
    "count": 228
  },
  {
    "gold": "NOUN",
    "prediction": "PROPN",
    "count": 119
  },
  {
    "gold": "ADJ",
    "prediction": "NOUN",
    "count": 102
  },
  {
    "gold": "NOUN",
    "prediction": "ADJ",
    "count": 78
  },
  {
    "gold": "AUX",
    "prediction": "PUNCT",
    "count": 58
  },
  {
    "gold": "ADJ",
    "prediction": "VERB",
    "count": 55
  },
  {
    "gold": "VERB",
    "prediction": "ADJ",
    "count": 50
  },
  {
    "gold": "ADV",
    "prediction": "ADJ",
    "count": 49
  },
  {
    "gold": "ADV",
    "prediction": "ADP",
    "count": 47
  },
  {
    "gold": "ADJ",
    "prediction": "PROPN",
    "count": 46
  }
]
```

## Errores partial_finetuning
{"words": 25094, "errors": 1071, "correct": 24023}
```json
[
  {
    "gold": "PROPN",
    "prediction": "NOUN",
    "count": 180
  },
  {
    "gold": "NOUN",
    "prediction": "PROPN",
    "count": 109
  },
  {
    "gold": "ADJ",
    "prediction": "NOUN",
    "count": 64
  },
  {
    "gold": "NOUN",
    "prediction": "ADJ",
    "count": 60
  },
  {
    "gold": "ADV",
    "prediction": "ADJ",
    "count": 35
  },
  {
    "gold": "ADJ",
    "prediction": "VERB",
    "count": 35
  },
  {
    "gold": "PROPN",
    "prediction": "ADJ",
    "count": 34
  },
  {
    "gold": "VERB",
    "prediction": "ADJ",
    "count": 33
  },
  {
    "gold": "ADV",
    "prediction": "ADP",
    "count": 30
  },
  {
    "gold": "ADJ",
    "prediction": "PROPN",
    "count": 26
  }
]
```

Fuentes: https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15 ; https://universaldependencies.org/format.html
Procedencia y SHA256: source_files.json. Entorno bloqueado, predicciones, historias y exportación dentro del directorio de ejecución.