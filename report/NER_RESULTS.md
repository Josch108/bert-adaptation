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
Las cuatro tareas y el reporte PDF final están completos. Modelos publicados: ver publication_manifest.json.

## Interpretación de esta ejecución

El modelo seleccionado encontró correctamente **5,189 de 5,648 entidades reales** y produjo 505 entidades falsas o con tipo/límites incorrectos. F1 no es el porcentaje de palabras correctas. Una entidad requiere coincidencia completa.

La diferencia observada de F1 en prueba entre completo y parcial es 2.39 puntos porcentuales. El ajuste completo empleó 3.03 veces el tiempo de entrenamiento del parcial. Esta comparación se limita a la configuración y datos utilizados; no es una conclusión universal.

**Cautela:** la diferencia está dentro del orden de 1–3 puntos que la consigna advierte como posible variación entre semillas. Con una sola semilla no podemos separar la ventaja del método de esa variabilidad. El ganador es una elección operativa según el criterio fijado, no una demostración de superioridad robusta. No se repitieron ejecuciones para buscar una ventaja.

### Comparación de convenciones de evaluación

| Método | F1 estricto IOB2 (principal) | F1 estilo conlleval (secundario) | Transiciones I inválidas |
|---|---:|---:|---:|
| partial_finetuning | 0.8911 | 0.8796 | 182 |
| full_finetuning | 0.9150 | 0.9104 | 64 |

Las convenciones interpretan secuencias BIO inválidas de forma diferente; no se deben comparar cifras sin indicar la convención. El modo estricto no tiene que producir siempre el menor F1. No se repararon etiquetas después de predecir.

| Tipo de entidad | Precisión | Recall | F1 | Entidades reales |
|---|---:|---:|---:|---:|
| PER | 0.9605 | 0.9623 | 0.9614 | 1617 |
| ORG | 0.8832 | 0.9103 | 0.8965 | 1661 |
| LOC | 0.9326 | 0.9293 | 0.9309 | 1668 |
| MISC | 0.8157 | 0.8134 | 0.8146 | 702 |

## Ejemplos de errores del ganador

Estos ejemplos describen errores de prueba; no se utilizaron para volver a ajustar el modelo.

- Frase: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
  - Entidades reales no acertadas: [{"type": "LOC", "start_word": 2, "end_word_exclusive": 3, "text": "JAPAN"}, {"type": "PER", "start_word": 7, "end_word_exclusive": 8, "text": "CHINA"}]
  - Predicciones no coincidentes: [{"type": "ORG", "start_word": 7, "end_word_exclusive": 8, "text": "CHINA"}, {"type": "PER", "start_word": 2, "end_word_exclusive": 3, "text": "JAPAN"}, {"type": "PER", "start_word": 4, "end_word_exclusive": 5, "text": "LUCKY"}]
- Frase: Defender Hassan Abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of Bitar 's goal .
  - Entidades reales no acertadas: [{"type": "PER", "start_word": 27, "end_word_exclusive": 28, "text": "Bitar"}]
  - Predicciones no coincidentes: [{"type": "ORG", "start_word": 27, "end_word_exclusive": 28, "text": "Bitar"}]
- Frase: Bitar pulled off fine saves whenever they did .
  - Entidades reales no acertadas: [{"type": "PER", "start_word": 0, "end_word_exclusive": 1, "text": "Bitar"}]
  - Predicciones no coincidentes: [{"type": "ORG", "start_word": 0, "end_word_exclusive": 1, "text": "Bitar"}]
- Frase: RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR .
  - Entidades reales no acertadas: [{"type": "LOC", "start_word": 6, "end_word_exclusive": 7, "text": "ITALY"}, {"type": "PER", "start_word": 3, "end_word_exclusive": 4, "text": "CUTTITTA"}]
  - Predicciones no coincidentes: [{"type": "ORG", "start_word": 3, "end_word_exclusive": 4, "text": "CUTTITTA"}]
- Frase: Cuttitta announced his retirement after the 1995 World Cup , where he took issue with being dropped from the Italy side that faced England in the pool stages .
  - Entidades reales no acertadas: [{"type": "MISC", "start_word": 6, "end_word_exclusive": 9, "text": "1995 World Cup"}]
  - Predicciones no coincidentes: [{"type": "MISC", "start_word": 7, "end_word_exclusive": 9, "text": "World Cup"}]

## Comprobaciones

Pasaron tres pruebas offline sobre alineación, métrica estricta, selección y entrenamiento de un BERT diminuto;
la prueba corta en GPU; y la ejecución completa de las seis celdas de código del notebook sin errores.
Se recalcularon las métricas desde todas las predicciones guardadas y se comprobó que las cuentas de entidades
reproducen F1. Los pesos exportados coinciden por SHA-256 con el checkpoint elegido; el código guardado coincide
con el ejecutado. La inferencia del modelo exportado se guarda en `inference_example.json`.

## Reproducibilidad y limitaciones

Una semilla y tres épocas por configuración. Se conservaron las particiones oficiales, incluyendo frases repetidas
entre ellas (ver auditoría arriba); no debe presentarse el conjunto como libre de duplicados textuales.
No hay truncamiento ni reparación automática de predicciones BIO inválidas.
Los checkpoints se eligen por F1 estricto de validación; el F1 conlleval secundario puede diferir por su tratamiento de BIO.
Los tiempos son de entrenamiento y excluyen validación, descarga y guardado; no miden la latencia de inferencia.
La precisión, versiones completas de dependencias, revisiones del modelo/dataset e índices están guardados en la ejecución.

## Archivos

- [Notebook ejecutado](../notebooks/02_ner_conll2003.ipynb)
- [Resultados CSV](../runs/ner_verified/comparison.csv)
- [Curvas](../runs/ner_verified/learning_curves.png)
- [Alineación del batch](../runs/ner_verified/alignment_check.csv)
- [Configuración](../runs/ner_verified/config.json)
- [Model card](../runs/ner_verified/delivered_model_ner/README.md)
- [Guía](../NER_GUIDE.md)

El modelo está exportado localmente, todavía sin publicación en Hugging Face.
POS, QA y el reporte PDF integrado siguen pendientes.
