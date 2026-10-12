# Validación de esta implementación

## Verificación ejecutada

- Instalación editable con extras de visor y pruebas: correcta.
- `python -m pytest -q`: **26 pruebas aprobadas**.
- Generación de ocho sujetos sintéticos: correcta.
- Entrenamiento de los cuatro modelos, umbrales con validación y evaluación en sujetos de prueba: correcto.
- Inferencia desde un modelo guardado sin cargar hipnogramas: correcta.
- HTTP de Dash: `/`, `/_dash-layout`, `/_dash-dependencies` responden 200.
- Callbacks de carga fallida, exploración multicanal y visor de probabilidades: comprobados.
- No se realizó inspección visual automatizada en navegador ni prueba con hardware de adquisición.

## Pruebas científicamente relevantes

- Sustituir muestras futuras no altera el filtrado pasado.
- Truncar el archivo en t conserva exactamente características y diferencias calculadas hasta t.
- Etiquetas respetan anticipación mínima y horizonte semiabierto.
- Vigilia sostenida fusiona épocas contiguas; huecos y etapas desconocidas rompen episodios.
- Alertas demasiado tardías no cuentan como anticipaciones.
- Refractario y falsas alarmas se comprueban con eventos conocidos.
- No tener eventos produce sensibilidad indefinida, no cero inventado.
- Sujetos compartidos entre particiones se rechazan.
- Huecos/NaN, timestamps irregulares, unidades faltantes y anotaciones solapadas se rechazan.
- Predicciones de inferencia y del test coinciden en instantes comunes.
- El final sin suficiente futuro anotado se excluye de evaluación; inferencia no necesita ese futuro.
- Sustitutos multicanal conservan autoespectros y espectros cruzados numéricamente.
- Ventanas planas no producen una dimensión artificialmente enorme.
- La interfaz respeta diez dimensiones de embedding para dos canales y m=5.

## Comprobación con archivo real del repositorio

Se cargó `demo/SC4001E0-PSG.edf` mediante MNE:

- canales: `EEG Fpz-Cz`, `EEG Pz-Oz`;
- muestreo: 100 Hz;
- muestras: 7 950 000 por canal;
- ventana explorada: 20–80 s;
- embedding: m=5 por canal, tau=5 muestras;
- matriz resultante: 5980 × 10;
- callback del visor terminó correctamente, con señal, PSD, proyección y diagnósticos.

Este ensayo comprueba lectura y análisis, **no** rendimiento de anticipación: no se utilizó un hipnograma para ese registro.

## Ensayo sintético reproducible

Protocolo predeterminado, cuatro sujetos de entrenamiento, dos de validación y dos de prueba. Cuatro eventos evaluables en prueba:

| Modelo | Sensibilidad | Falsas alarmas/h | Brier aproximado |
|---|---:|---:|---:|
| Prevalencia | 0 | 0 | 0.158 |
| Espectral | 1 | 0 | 0.022 |
| Individual | 1 | 0 | 0.025 |
| Conjunto | 1 | 0 | 0.024 |

La señal sintética contiene cambios predespertar introducidos deliberadamente. Estos valores solo verifican que el flujo funciona y que los modelos de referencia se reportan de manera transparente. Aquí la geometría **no mejora** la sensibilidad y el espectro obtiene menor Brier. No son resultados clínicos, no prueban un atractor y no anticipan el rendimiento en EEG real.

## Entorno comprobado

Python 3.12; numpy 2.3.5; scipy 1.17.0; pandas 2.2.3; scikit-learn 1.8.0; joblib 1.5.3; MNE 1.13.2; Dash 4.4.1; dash-bootstrap-components 2.0.4; Plotly 7.1.0; pytest 9.1.1. Cada experimento guarda las versiones del motor en report.json. Versiones diferentes pueden cambiar resultados numéricos ligeramente.

La CI configura Python 3.11 y 3.12; su estado remoto debe consultarse tras publicar la rama. Su configuración no equivale a una ejecución remota ya completada.
