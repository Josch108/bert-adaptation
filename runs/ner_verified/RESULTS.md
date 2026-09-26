# NER: resultados medidos

Comparación final; una semilla por configuración.

Particiones oficiales: {'train': 14041, 'validation': 3250, 'test': 3453}. Sin truncamiento; máximos de subtokens: {'train': 173, 'validation': 151, 'test': 148}.
Frases exactas compartidas entre particiones (conservadas para mantener el benchmark): {'train/validation': 129, 'train/test': 78, 'validation/test': 25}.
Selección: F1 micro de entidades, seqeval strict IOB2. La entidad debe coincidir en tipo, inicio y fin.

| Método | Época | F1 validación | Precisión prueba | Recall prueba | F1 prueba | Entrenamiento (s) |
|---|---:|---:|---:|---:|---:|---:|
| partial_finetuning | 3 | 0.9204 | 0.8910 | 0.8913 | 0.8911 | 145.8 |
| full_finetuning | 3 | 0.9484 | 0.9113 | 0.9187 | 0.9150 | 441.9 |

Ganador por validación: **full_finetuning**. Prueba se evaluó después de guardar la selección.
Los tiempos excluyen validación y guardado; ambos métodos ejecutan el cuerpo de BERT en cada lote.
La pérdida se promedia por palabras supervisadas, no por frases ni subtokens ignorados.
Una semilla no establece superioridad estadística; diferencias pequeñas pueden depender del azar.
No se ajustaron hiperparámetros con prueba. Las etiquetas O no dominan la métrica principal de entidades.

## Errores: partial_finetuning

```json
{
  "gold_entities": 5648,
  "predicted_entities": 5650,
  "correct_entities": 5034,
  "false_positive_entities": 616,
  "wrong_type_exact_boundary": 240,
  "missed_entity": 160,
  "overlapping_wrong_boundary": 214
}
```

Tipos incorrectos, límites incorrectos y entidades omitidas son categorías excluyentes de errores sobre entidades reales. Falsos positivos se cuentan aparte.

## Errores: full_finetuning

```json
{
  "gold_entities": 5648,
  "predicted_entities": 5694,
  "correct_entities": 5189,
  "false_positive_entities": 505,
  "wrong_type_exact_boundary": 220,
  "missed_entity": 82,
  "overlapping_wrong_boundary": 157
}
```

Tipos incorrectos, límites incorrectos y entidades omitidas son categorías excluyentes de errores sobre entidades reales. Falsos positivos se cuentan aparte.

Evidencia: data_manifest.json, alignment_check.csv, historiales, selection.json, predicciones por palabra, métricas por tipo y ejemplos de error.
BERT-base-cased, precisión bf16 cuando la GPU lo admite, semilla 42 y dos grupos de LR (cabeza 1e-3, encoder 2e-5).
Fuentes: https://huggingface.co/datasets/lhoestq/conll2003 ; https://github.com/chakki-works/seqeval ; https://arxiv.org/abs/1810.04805
POS y QA siguen pendientes. El reporte PDF anterior aún no fue actualizado. Modelo exportado localmente, no publicado.
