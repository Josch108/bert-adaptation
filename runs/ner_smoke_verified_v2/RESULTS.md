# NER: resultados medidos

PRUEBA TECNICA: no son resultados finales.

Particiones oficiales: {'train': 32, 'validation': 16, 'test': 16}. Sin truncamiento; máximos de subtokens: {'train': 58, 'validation': 53, 'test': 37}.
Frases exactas compartidas entre particiones (conservadas para mantener el benchmark): {'train/validation': 0, 'train/test': 0, 'validation/test': 0}.
Selección: F1 micro de entidades, seqeval strict IOB2. La entidad debe coincidir en tipo, inicio y fin.

| Método | Época | F1 validación | Precisión prueba | Recall prueba | F1 prueba | Entrenamiento (s) |
|---|---:|---:|---:|---:|---:|---:|
| partial_finetuning | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4 |
| full_finetuning | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4 |

Ganador por validación: **partial_finetuning**. Prueba se evaluó después de guardar la selección.
Los tiempos excluyen validación y guardado; ambos métodos ejecutan el cuerpo de BERT en cada lote.
La pérdida se promedia por palabras supervisadas, no por frases ni subtokens ignorados.
Una semilla no establece superioridad estadística; diferencias pequeñas pueden depender del azar.
No se ajustaron hiperparámetros con prueba. Las etiquetas O no dominan la métrica principal de entidades.

## Errores: partial_finetuning

```json
{
  "gold_entities": 22,
  "predicted_entities": 2,
  "correct_entities": 0,
  "false_positive_entities": 2,
  "missed_entity": 21,
  "overlapping_wrong_boundary": 1
}
```

Tipos incorrectos, límites incorrectos y entidades omitidas son categorías excluyentes de errores sobre entidades reales. Falsos positivos se cuentan aparte.

## Errores: full_finetuning

```json
{
  "gold_entities": 22,
  "predicted_entities": 1,
  "correct_entities": 0,
  "false_positive_entities": 1,
  "missed_entity": 22
}
```

Tipos incorrectos, límites incorrectos y entidades omitidas son categorías excluyentes de errores sobre entidades reales. Falsos positivos se cuentan aparte.

Evidencia: data_manifest.json, alignment_check.csv, historiales, selection.json, predicciones por palabra, métricas por tipo y ejemplos de error.
BERT-base-cased, precisión bf16 cuando la GPU lo admite, semilla 42 y dos grupos de LR (cabeza 1e-3, encoder 2e-5).
Fuentes: https://huggingface.co/datasets/lhoestq/conll2003 ; https://github.com/chakki-works/seqeval ; https://arxiv.org/abs/1810.04805
POS y QA siguen pendientes. El reporte PDF anterior aún no fue actualizado. Modelo exportado localmente, no publicado.
