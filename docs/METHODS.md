# Protocolo de investigación

## Alcance e hipótesis

La comparación primaria evalúa si el embedding multicanal añade información predictiva al espectro. No intenta certificar un atractor ni inferir conexiones anatómicas. No incorpora cálculo exterior: sería una investigación posterior con hipótesis y controles propios.

H0 operacional: bajo este protocolo, añadir las características conjuntas no mejora la anticipación respecto al modelo espectral. No se promete mejora ni significación estadística.

## Cronología

Para una decisión t, la ventana es [t-window_s, t). El filtro SOS Butterworth es causal (`sosfilt`), se inicializa en cero al inicio de la grabación y no se reinicia por ventana. El resultado se calcula sobre el archivo completo, pero la recurrencia del filtro usa solo entradas pasadas: existe una prueba que sustituye todo el futuro y exige igualdad de las salidas previas. No se utilizan `filtfilt`, detrend global, interpolación con futuro ni decimación de fase cero.

Se conserva la frecuencia original. `high_hz` debe ser estrictamente menor que Nyquist. El retardo en muestras es `max(1, round(tau_s*fs))`; una tasa baja puede no representar exactamente el retardo solicitado. La ventana y el paso deben corresponder a números enteros de muestras. El retardo de fase del filtro es parte de la latencia efectiva, no se resta usando información futura.

El pasa-banda predeterminado 0.5–35 Hz limita deriva y contenido alto para una primera referencia de sueño con señales de 100 Hz. No es un requisito de Takens. La potencia llamada `low_gamma` solo cubre la porción 30–high_hz disponible. El orden 4 es el parámetro de diseño Butterworth: la transformación pasa-banda duplica el orden final. La duración de calentamiento de 20 s es una elección documentada, no una garantía para filtros arbitrarios; inspecciona la respuesta al modificar cortes/orden.

## Etiquetas y censura

Se fusionan etapas contiguas N1/N2/N3/REM en un episodio de sueño. Se fusionan épocas W contiguas en vigilia. Huecos, etapas desconocidas, movimiento y otras etiquetas rompen esos episodios.

Un evento requiere sueño continuo suficiente seguido inmediatamente por vigilia continua suficiente. El instante del evento es el inicio de W, aunque confirmar su duración requiere observación futura; esa observación se usa únicamente para la etiqueta.

`label=1` si un evento comienza en [t+lead_s, t+lead_s+horizon_s). El límite izquierdo está incluido y el derecho excluido. Un evento más cercano que lead_s no es una anticipación válida para esa decisión.

Solo se evalúan ventanas con anotaciones conocidas desde t-window_s hasta t+lead_s+horizon_s+min_wake_s, y señal disponible hasta ese límite. Esto excluye el final censurado y periodos no anotados. La inferencia no usa etiquetas y sí puede emitir decisiones hasta el final del archivo.

Las alertas se evalúan tanto durante sueño como durante vigilia. No hay gate que consulte la etiqueta experta actual. El tiempo evaluado excluye huecos, inicio de calentamiento y final censurado; por tanto, la tasa reportada no debe presentarse como falsas alarmas por hora de una noche completa sin esa aclaración.

## Representación y características

Para C canales y m bloques, X_n concatena [y_n, y_(n-tau), ..., y_(n-(m-1)tau)] en R^(C*m). Las columnas conservan el orden especificado en el manifiesto. No se limita el cálculo a tres dimensiones.

El espectro se estima con Welch y segmentos de hasta 4 s. Cada banda se integra numéricamente y divide por la suma de las bandas definidas. Las bandas adyacentes usan límites semiabiertos para no contar dos veces bins de frontera. El log RMS en µV retiene amplitud absoluta.

Para geometría se centra y escala cada canal usando únicamente la ventana pasada. La magnitud queda en las características espectrales; la normalización reduce la dominancia de un canal por escala. No garantiza independencia de referencia o mezcla de fuentes. Ventanas con algún canal constante fallan explícitamente en vez de generar valores arbitrarios.

Las distancias se dividen por sqrt(C*m). Se usan como máximo 256 puntos uniformemente distribuidos por ventana para limitar coste. La exclusión de Theiler utiliza los índices originales, no el índice reducido. Se calculan:

- rango efectivo de la covarianza por participación de sus autovalores;
- fracción de varianza de la primera componente;
- mediana de distancias de pares permitidos;
- tasa de recurrencia con radio normalizado fijo;
- persistencia de recurrencias en puntos consecutivos de la muestra reducida.

La última medida es una proporción condicional, **no** el DET completo de RQA. Su intervalo depende del muestreo de puntos y la longitud de ventana. Todas las familias añaden diferencias respecto a la decisión anterior; la primera diferencia es cero. El protocolo de inferencia debe mantener el paso y longitud de ventana del entrenamiento.

El visor incorpora AMI, FNN con Theiler y curvas de pendiente local de correlación. No devuelve automáticamente un D2 verdadero o una etiqueta de caos. Su PCA es solo una proyección visual local y no se utiliza en los modelos.

## Datos sustitutos

`awakening.diagnostics.phase_surrogate` aplica las mismas fases aleatorias de Fourier a todos los canales. Preserva autoespectros y espectros cruzados, no los histogramas marginales. Es un control de estructura espectral lineal estacionaria, no un test universal de determinismo. La prueba automatizada verifica la preservación espectral.

No hay un contraste de hipótesis completo ni corrección de comparaciones múltiples incorporados. Antes de una prueba con sustitutos, fija estadístico, hipótesis nula, número de réplicas, manejo de no estacionariedad y unidad de análisis. No uses sustitutos de forma independiente por canal para concluir que cualquier diferencia demuestra acoplamiento no lineal: eso destruiría también estructura lineal conjunta.

## Entrenamiento y selección

Particiones explícitas, disjuntas por sujeto. El escalador y la regresión se ajustan solo a entrenamiento. El hiperparámetro C=1 se fija por adelantado. No se optimizan embedding, filtros o selección de canales con test. Si se exploran esos parámetros después, usa validación interna por sujetos y conserva una prueba externa sin mirar.

La regresión logística no utiliza `class_weight=balanced`, porque cambia el prior implícito. Se reportan Brier y tablas de confiabilidad; esto evalúa calibración sin afirmar que esté conseguida. No se realiza calibración adicional con el mismo conjunto usado para estimar rendimiento final.

El modelo constante expone el efecto de desbalance. Las tres variantes entrenables comparten clasificación, escalamiento y regla de selección de umbral. No se adjudica un ganador automáticamente en función del test.

## Alertas y métricas

Se emite alerta cuando p >= umbral y ha terminado el intervalo refractario anterior. El refractario dura lead_s+horizon_s. Un riesgo continuamente alto puede volver a generar alertas después de cada refractario, por lo que no obtiene crédito ilimitado por una única alerta al comienzo de la noche.

Una alerta emitida a t apunta a [t+lead, t+lead+horizon). Es falsa si ningún evento evaluable comienza allí. Un evento cuenta como detectado una sola vez y la anticipación se mide desde su primera alerta válida. Solo se cuentan eventos para los que existe al menos una decisión evaluada que podría anticiparlos.

El tiempo en alerta es la duración de los intervalos objetivo dentro de la unión de intervalos puntuados [t,t+step). No incluye el hueco de anticipación mínima. Las horas puntuadas son número de decisiones por step/3600. Reportar este denominador evita contar horas excluidas por falta de etiquetas.

El umbral se elige entre 0..1 en pasos .01 y un umbral mayor que 1 (no emitir). Entre opciones que cumplen el presupuesto de falsas alarmas en validación se maximiza sensibilidad, luego se minimizan falsas alarmas y tiempo en alerta. El test utiliza el mismo umbral bloqueado. No hay garantía de mantener el presupuesto en una nueva población.

Se reportan intervalos bootstrap percentiles por sujeto y una comparación pareada de diferencia de sensibilidad joint-spectral. No son una corrección de múltiples comparaciones ni un test confirmatorio por sí solos. Con un solo sujeto no se presenta intervalo; con pocos sujetos puede ser poco informativo. Average precision y ROC-AUC son undefined si solo hay una clase. Sensibilidad es undefined si no hay eventos evaluables.

## Próximo experimento real

1. Reunir PSG e hipnogramas de múltiples sujetos, registrar población, medicación y montaje.
2. Comprobar sincronización, unidades, referencia y calidad. No mezclar cohorts sin documentarlo.
3. Fijar particiones, filtros y evento antes de explorar resultados de prueba.
4. Ejecutar la comparación espectral/individual/conjunta, inspeccionar artefactos y falsas alarmas.
5. Repetir en otra cohorte o dataset para estudiar generalización.

Una demostración de anticipación en humanos requiere estos datos y su evaluación. El código y los ejemplos sintéticos no sustituyen ese resultado.

## Referencias metodológicas

- Deyle & Sugihara (2011). Generalized theorems for nonlinear state space reconstruction. https://doi.org/10.1371/journal.pone.0018295
- Theiler et al. (1992). Testing for nonlinearity in time series: the method of surrogate data. https://doi.org/10.1016/0167-2789(92)90102-S
- Widmann et al. (2015). Digital filter design for electrophysiological data—a practical approach. https://doi.org/10.1016/j.jneumeth.2014.08.002
- Sleep-EDF Expanded: https://physionet.org/content/sleep-edfx/
