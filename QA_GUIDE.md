# Guía QA

QA extractivo localiza una respuesta ya escrita. En esta etapa comparamos adaptar dos capas finales versus adaptar BERT completo.

## Archivos
- src/qa_experiment.py: experimento y métricas.
- notebooks/04_extractive_qa_squad.ipynb: explicación y lectura ejecutada de la evidencia real; no simula reentrenamiento.
- tests/test_qa.py: alineación en ventanas, tokens de contexto, múltiples respuestas y decodificación.
- runs/qa_verified: evidencia completa y delivered_model_qa.
- report/QA_RESULTS.md: resultados y errores.

## Reproducir
Usar el entorno de las etapas anteriores; las versiones exactas están en runs/qa_verified/requirements-lock.txt. Torch con CUDA requiere índice compatible con el equipo. Desde la raíz:

```powershell
python -m unittest discover -s tests -p test_qa.py -v
python src/qa_experiment.py --output runs/qa_smoke_new --smoke
python src/qa_experiment.py --output runs/qa_new
```

Para reconstruir exactamente la configuración y revisiones guardadas, ejecutar desde la raíz un archivo Python con:

```python
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
from qa_experiment import Config, run
cfg = Config(**json.loads(Path('runs/qa_verified/config.json').read_text(encoding='utf-8')))
run(cfg, 'runs/qa_reproduced')
```

La CLI por defecto resuelve revisiones actuales; este bloque fija las mismas revisiones del experimento. Semilla fija no garantiza identidad bit a bit porque SDPA backward puede ser no determinista.

Una ruta existente con experimento no se sobrescribe. Smoke usa 32/16/16 preguntas y 1 época; verifica el flujo y no entrega un modelo final. --prepare-only descarga/tokeniza sin entrenamiento. En esta computadora se utilizaron HF_HOME y el bundle de certificados del entorno previo; no se desactiva TLS. No se ejecutan POS y QA simultáneamente en GPU para evitar memoria insuficiente.

El aviso de Transformers sobre pesos qa_outputs recién inicializados es esperado: BERT aporta el cuerpo preentrenado y la cabeza de inicio/final se aprende con esta tarea.

Interpretación: EM requiere coincidencia normalizada completa; F1 admite respuestas con palabras parcialmente compartidas. Una respuesta como “London, England” frente a “London” puede tener F1 parcial pero EM cero. El checkpoint se elige con F1 de validación interna, nunca prueba. La pérdida se computa por ventana; las métricas por pregunta.
