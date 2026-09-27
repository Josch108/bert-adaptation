# AG News: ejecución reproducible

El notebook `notebooks/01_topic_classification_agnews.ipynb` explica y ejecuta la comparación.
La implementación compartida está en `src/agnews_experiment.py`.

## Instalación

Usar Python 3.12 y un entorno virtual. En Windows con NVIDIA:

```powershell
python -m venv .venv-agnews
.\.venv-agnews\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv-agnews\Scripts\python.exe -m pip install -r requirements-agnews.txt
```

En Colab se puede instalar `requirements-agnews.txt` con `%pip` y reiniciar el kernel antes de importar.
Es necesario conservar la carpeta `src` junto a `notebooks`.

## Prueba corta y entrenamiento final

Desde la raíz del proyecto, con el entorno activado:

```powershell
python src/agnews_experiment.py --smoke --output runs/smoke_01
python src/agnews_experiment.py --output runs/agnews_01
python src/agnews_experiment.py --config runs/agnews_01/config.json --output runs/agnews_replicated
```

Cada salida debe ser una carpeta que todavía no exista. La prueba corta NO aporta métricas finales.
Alternativa: ejecutar todas las celdas del notebook usando el kernel del entorno instalado.
En este equipo, el entorno preparado por Codex está en
`C:\Users\IGNITER\Documents\ChatGPT\Cosas Escuela 2\.venv-agnews`.

## Decisiones metodológicas

- Misma cabeza lineal y pooler en ambas alternativas: permite aislar mejor congelado vs. ajuste completo.
- Train/validation/test separados, estratificados, sin duplicados de texto exactos normalizados entre grupos.
- 12,000/2,000/2,000 filas. Los índices originales, exclusiones y commits del modelo/dataset quedan en `splits.json`.
- Tres épocas, semilla 42, máximo 128 tokens; microbatch 8 x acumulación 4 = batch efectivo 32.
- El extractor congelado mantiene dropout desactivado. Se verifica que los pesos de BERT no cambien.
- Se elige la mejor época y alternativa con macro-F1 de validación, nunca con prueba.
- Empates entre épocas conservan la primera; empate entre alternativas favorece menos parámetros entrenables.
- Se guardan los mejores modelos de ambas alternativas y se evalúan en prueba después de fijar el ganador.
- Una ejecución por configuración. No se afirma significancia estadística ni una explicación causal de las diferencias.
- Precisión bf16 si la GPU lo admite; de lo contrario fp32. No se prometen resultados idénticos entre hardware/versiones.

## Evidencia

Cada ejecución contiene configuración, entorno, particiones, historial JSON/CSV, selección,
matrices de confusión, todas las predicciones de prueba, curvas y `RESULTS.md`.
Una ejecución completa produce además `delivered_model_agnews/`, compatible con `pipeline` de Transformers.
No se publica automáticamente. La model card no afirma accesos que aún no se hayan comprobado.

El reporte final en inglés incorpora los resultados verificados de `runs/`; el borrador anterior está archivado.
NER, POS y QA no han sido reentrenados en esta etapa.

## Comprobaciones automáticas

`python tests/test_agnews.py` verifica offline, usando un BERT diminuto aleatorio, que:
la cabeza aprende, el cuerpo cambia solo en ajuste completo, los checkpoints se recargan,
y la selección depende de validación aunque prueba favorezca al otro modelo.
Estas pruebas verifican código, no el rendimiento de BERT-base.
