# EEG: espacio de estados y anticipación de despertares

Herramienta de investigación para explorar EEG multicanal y comprobar si su geometría temporal mejora la anticipación de despertares respecto a un modelo espectral. Incluye un visor Dash y un flujo reproducible de entrenamiento, evaluación e inferencia.

**Estado:** implementación comprobada con pruebas y datos sintéticos. No incluye un modelo validado en personas. Una trayectoria reconstruida no demuestra un atractor, caos ni un diagnóstico.

## Inicio rápido

Requiere Python 3.10 o posterior. Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dashboard,test]'
python -m pytest -q
python atractor.py
```

Abre http://127.0.0.1:8050. Para entrenar sin interfaz basta `python -m pip install -e '.[eeg]'`.

El visor permite seleccionar varios canales, un retardo fijo y dimensiones de 2 a 12 por canal. El cálculo conserva todas las coordenadas; la imagen muestra una proyección PCA de tres dimensiones ajustada para esa ventana. El color indica tiempo. Cambiar ventana reajusta esa proyección: sus ejes no son comparables entre figuras.

La pestaña **Probabilidad de despertar** abre los CSV de probabilidades generados por la CLI. El entrenamiento se ejecuta mediante comandos, no dentro de un callback de la interfaz.

## Demostración completa

```bash
python -m awakening demo --output data/synthetic
python -m awakening train \
  --manifest data/synthetic/manifest.json \
  --output results/synthetic
python -m awakening predict \
  --model results/synthetic/joint.joblib \
  --signal data/synthetic/s06.npz \
  --output results/synthetic/inference.csv
```

Los directorios de demo y entrenamiento deben ser nuevos: no se sobrescriben resultados anteriores. La demo genera ocho sujetos artificiales, dos canales y cambios predespertar introducidos deliberadamente. Sirve para probar el software, **no para medir rendimiento fisiológico**. Ni los registros generados ni los modelos entrenados se suben a Git.

## Pregunta y protocolo predeterminado

> ¿Las características de una reconstrucción multicanal añaden información predictiva a las potencias espectrales para anticipar un despertar sostenido?

- Evento: inicio de una vigilia continua de al menos 60 s, inmediatamente después de al menos 120 s de sueño continuo.
- Observación: últimos 60 s, una decisión cada 10 s.
- Objetivo: inicio del evento en **[t + 30 s, t + 150 s)**. Hay 30 s de anticipación mínima y una ventana objetivo de 120 s.
- Filtrado: Butterworth causal, orden de diseño 4, 0.5–35 Hz, sin decimación; se descartan 20 s iniciales antes de comenzar la primera ventana.
- Embedding: cuatro bloques de coordenadas por canal, retardo nominal 50 ms, Theiler de 1 s.
- Se puntúan todas las ventanas válidas, **incluida vigilia**. No se usa la etiqueta experta del estado actual para habilitar alertas.
- Los umbrales de alerta se seleccionan solo con sujetos de validación, con presupuesto predeterminado de 0.5 falsas alarmas por hora evaluada.

Son decisiones iniciales configurables, no parámetros clínicamente óptimos. El horizonte no significa que se pronostiquen exactamente dos minutos antes del evento. El mínimo y máximo de aviso son 30 y menos de 150 segundos, respectivamente.

## Datos reales

Consulta [examples/manifest.example.json](examples/manifest.example.json) y [docs/METHODS.md](docs/METHODS.md).

El manifiesto define un orden fijo de canales y cada noche con su identificador, sujeto, partición, señal y anotaciones. Todas las noches del mismo sujeto deben permanecer en la misma partición. El programa rechaza duplicados de archivos y sujetos entre particiones.

```json
{
  "channels": ["EEG Fpz-Cz", "EEG Pz-Oz"],
  "recordings": [
    {
      "id": "subject01_night1",
      "subject": "subject01",
      "split": "train",
      "signal": "subject01-PSG.edf",
      "annotations": "subject01-Hypnogram.edf",
      "annotation_offset_s": 0
    }
  ]
}
```

Este fragmento muestra un registro: un experimento necesita las tres particiones `train`, `validation` y `test`, con sujetos diferentes, positivos y negativos en entrenamiento y eventos evaluables en validación. Conviene disponer de múltiples sujetos por partición para estimar incertidumbre. No derives IDs de sujeto diferentes de noches de una misma persona.

Formatos:

| Datos | Formato y contrato |
|---|---|
| EEG y otros canales de voltaje | EDF/BDF/FIF/SET mediante MNE; unidades internas V. Selecciona canales y verifica su montaje y referencia. |
| NPZ | `data`: muestras × canales; `channels`: nombres; `fs`: Hz; `unit`: `V` o `uV`. Sin objetos pickle. |
| CSV/TSV/TXT | Columnas seleccionadas por nombre; declarar `fs` y `unit`, o una `time_column` regular en segundos. |
| NPY | Muestras × canales en el orden del manifiesto; declarar `fs` y `unit`. No se adivina la orientación. |
| Hipnograma CSV | `onset,duration,stage`, segundos desde el inicio de la señal. W, N1, N2, N3, REM. |
| Hipnograma EDF+ | Anotaciones Sleep-EDF; mapea R a REM y etapas 3/4 a N3. |

La CLI permite archivos SET con sus archivos auxiliares presentes junto al original. El visor acepta SET autocontenidos; para un SET que dependa de FDT usa la CLI o convierte antes.

No se borran huecos ni se interpolan silenciosamente datos. Señales con NaN, relojes irregulares o ventanas planas detienen el análisis con un error. Las anotaciones desconocidas o incompletas excluyen ventanas de evaluación. Comprueba expresamente la alineación temporal entre PSG e hipnograma.

El repositorio contiene `demo/SC4001E0-PSG.edf`, pero no su hipnograma. Una sola noche sin etiquetas no permite entrenar y probar este objetivo. Sleep-EDF Expanded ofrece registros y sus hipnogramas correspondientes: https://physionet.org/content/sleep-edfx/ . Sus etiquetas de sueño/vigilia no equivalen a anotaciones específicas de microdespertares.

## Entrenar y comparar

```bash
python -m awakening train \
  --manifest data/my-study/manifest.json \
  --config examples/config.json \
  --false-alarms-per-hour 0.5 \
  --output results/my-study-v1
```

Se ejecutan cuatro modelos preespecificados:

1. `prevalence`: probabilidad constante estimada en entrenamiento.
2. `spectral`: potencias relativas por banda, log RMS y cambios respecto a la decisión anterior.
3. `individual`: espectro más geometría de cada canal por separado.
4. `joint`: espectro más geometría del embedding multicanal conjunto.

Los modelos 2–4 son regresiones logísticas con escalamiento ajustado **solo en entrenamiento**, sin ponderación de clases que cambie artificialmente las probabilidades. Es una referencia interpretable, no una afirmación de que la regresión logística sea el mejor modelo posible. El modelo temporal de estados de sueño discutido durante el diseño no está incluido: requeriría definir cómo obtener estados pasados sin etiquetas expertas durante inferencia.

Salidas:

- `summary.md`, `report.json`: métricas, configuración, versiones, hashes de archivos y comparación conjunta frente a espectral.
- `features.csv`, `events.csv`: características y eventos, conservando sujeto, noche y partición.
- `*_validation_thresholds.csv`: exploración de umbrales realizada únicamente en validación.
- `*_test_predictions.csv`, `*_test_alerts.csv`: probabilidades y alertas en prueba.
- `*_test_recordings.csv`, `*_test_reliability.csv`: detalle por registro y calibración por intervalos.
- `*.joblib`: modelos con orden de canales/características, configuración y umbral guardados.

La restricción de falsas alarmas se aplica en validación y puede incumplirse en sujetos nuevos. El informe muestra sensibilidad por evento, falsas alarmas/h, tiempo en alerta, anticipación, Brier, average precision, ROC-AUC y bootstrap por sujeto. No se selecciona el mejor modelo viendo el test. La comparación pareada principal es `joint` menos `spectral`; debe leerse junto a sus falsas alarmas.

## Inferencia sin hipnograma

```bash
python -m awakening predict \
  --model results/my-study-v1/joint.joblib \
  --signal data/new-night.edf \
  --output results/new-night-predictions.csv
```

Para CSV/NPY añade `--fs` y `--unit`; para CSV con tiempo también puedes usar `--time-column`. El orden de canales lo determina el modelo. Una frecuencia de muestreo no observada durante entrenamiento se rechaza; requiere remuestreo causal documentado o reentrenamiento. Carga solo modelos joblib de procedencia confiable: este formato ejecuta deserialización Python.

La inferencia simula decisiones causales sobre un archivo. Las características de cada decisión dependen exclusivamente de su pasado. No incluye adquisición desde un amplificador en vivo, notificaciones o actuación. Evalúa canales, referencia, población y muestreo compatibles con entrenamiento antes de interpretar probabilidades.

## Estructura

- `awakening/io.py`, `labels.py`: datos y definición de eventos.
- `features.py`, `diagnostics.py`: filtros, embedding, geometría, AMI, FNN y controles espectrales.
- `dataset.py`: ventanas y procedencia; censura del final del registro.
- `evaluation.py`, `model.py`: comparación, umbrales, métricas y entrenamiento.
- `cli.py`, `demo.py`: comandos y ejemplo completo.
- `atractor.py`: visor multicanal y resultados de pronóstico.
- `tests/`: causalidad, sincronización, eventos, particiones, inferencia y callbacks.

Ver [métodos y limitaciones](docs/METHODS.md) y [validación realizada](docs/VALIDATION.md). Los PDF y la captura antigua incluidos en el repositorio describen versiones anteriores y no son documentación de este protocolo.
