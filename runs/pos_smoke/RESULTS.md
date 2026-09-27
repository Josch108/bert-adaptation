# POS: resultados medidos

Prueba técnica; no final.
Particiones oficiales UD English EWT r2.15: {'train': 32, 'validation': 16, 'test': 16}.
Máximos de subtokens: {'train': 63, 'validation': 52, 'test': 68}. Sin truncamiento.
Frases repetidas entre particiones, conservadas: {'train/validation': 0, 'train/test': 0, 'validation/test': 0}.
Selección por accuracy de palabras en validación; empate entre métodos: menos parámetros entrenables.
Prueba se evalúa después de guardar la selección. Macro-F1 incluye las 17 etiquetas UPOS.

| Método | Época | Accuracy validación | Accuracy prueba | Macro-F1 prueba | Entrenamiento s |
|---|---:|---:|---:|---:|---:|
| frozen | 1 | 0.1000 | 0.0854 | 0.0722 | 0.9 |
| partial_finetuning | 1 | 0.1105 | 0.1057 | 0.0788 | 0.1 |

Seleccionado: **partial_finetuning**.
Tiempos excluyen validación y guardado; no son latencia de inferencia.
La pérdida se pondera por palabras supervisadas. Solo el primer subtoken recibe etiqueta.
Las filas multipalabra y nodos vacíos de CoNLL-U no son palabras sintácticas y se omiten; etiquetas desconocidas abortan la carga.
Modelo base uncased: puede limitar distinciones apoyadas en mayúsculas como PROPN/NOUN.
Una semilla no demuestra superioridad estadística. Sin búsqueda de hiperparámetros usando prueba.

## Errores frozen
{"words": 246, "errors": 225, "correct": 21}
```json
[
  {
    "gold": "PUNCT",
    "prediction": "CCONJ",
    "count": 12
  },
  {
    "gold": "VERB",
    "prediction": "INTJ",
    "count": 10
  },
  {
    "gold": "NOUN",
    "prediction": "AUX",
    "count": 8
  },
  {
    "gold": "PROPN",
    "prediction": "ADJ",
    "count": 7
  },
  {
    "gold": "PROPN",
    "prediction": "NOUN",
    "count": 7
  },
  {
    "gold": "DET",
    "prediction": "NOUN",
    "count": 7
  },
  {
    "gold": "VERB",
    "prediction": "AUX",
    "count": 7
  },
  {
    "gold": "ADV",
    "prediction": "NOUN",
    "count": 6
  },
  {
    "gold": "ADJ",
    "prediction": "NOUN",
    "count": 6
  },
  {
    "gold": "PUNCT",
    "prediction": "AUX",
    "count": 6
  }
]
```

## Errores partial_finetuning
{"words": 246, "errors": 220, "correct": 26}
```json
[
  {
    "gold": "PUNCT",
    "prediction": "CCONJ",
    "count": 11
  },
  {
    "gold": "VERB",
    "prediction": "INTJ",
    "count": 9
  },
  {
    "gold": "NOUN",
    "prediction": "AUX",
    "count": 8
  },
  {
    "gold": "PROPN",
    "prediction": "ADJ",
    "count": 7
  },
  {
    "gold": "PROPN",
    "prediction": "NOUN",
    "count": 7
  },
  {
    "gold": "DET",
    "prediction": "NOUN",
    "count": 7
  },
  {
    "gold": "ADJ",
    "prediction": "NOUN",
    "count": 7
  },
  {
    "gold": "ADV",
    "prediction": "NOUN",
    "count": 6
  },
  {
    "gold": "PUNCT",
    "prediction": "AUX",
    "count": 6
  },
  {
    "gold": "ADP",
    "prediction": "NOUN",
    "count": 5
  }
]
```

Fuentes: https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15 ; https://universaldependencies.org/format.html
Procedencia y SHA256: source_files.json. Entorno bloqueado, predicciones, historias y exportación dentro del directorio de ejecución.