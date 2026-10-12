"""Local multichannel explorer. Training: python -m awakening --help."""
import base64
from collections import OrderedDict
import io
import json
from pathlib import Path
import tempfile
import threading
import uuid

import dash
from dash import Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal

from awakening.config import Config
from awakening.io import load_recording
from awakening.features import delay_embedding, filter_causal, geometry
from awakening.diagnostics import ami_delay, correlation_curve, fnn_curve

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "EEG · State space & awakening forecasts"
CACHE = OrderedDict()
CACHE_LOCK = threading.Lock()


def ingest_file(content_bytes, filename, fs=None, unit=None, time_column=None):
    """Read into volts. Text/NPY need explicit metadata; EDF-family uses its header."""
    suffix = Path(filename).suffix.lower()
    if len(content_bytes) > 64*1024*1024:
        raise ValueError("Use the CLI for files larger than 64 MiB")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ("signal" + suffix)
        path.write_bytes(content_bytes)
        if suffix in {".edf", ".bdf", ".fif", ".set"}:
            import mne
            readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
                       ".fif": mne.io.read_raw_fif, ".set": mne.io.read_raw_eeglab}
            raw = readers[suffix](str(path), preload=False, verbose="ERROR")
            channels = [n for n, t in zip(raw.ch_names, raw.get_channel_types()) if t in {"eeg", "eog", "emg"}]
            raw.close()
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                channels = z["channels"].astype(str).tolist()
        elif suffix == ".npy":
            a = np.load(path, allow_pickle=False)
            if a.ndim not in (1, 2):
                raise ValueError("NPY must have samples x channels")
            channels = [f"CH_{i}" for i in range(1 if a.ndim == 1 else a.shape[1])]
        elif suffix in {".csv", ".tsv", ".txt"}:
            df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else None, engine="python")
            channels = [str(c) for c in df.columns if c != time_column]
        else:
            raise ValueError("Supported: EDF/BDF/FIF/SET, CSV/TSV/TXT, NPY/NPZ")
        return load_recording(path, channels, fs, unit, time_column)


def store_recording(recording):
    token = uuid.uuid4().hex
    with CACHE_LOCK:
        CACHE[token] = recording
        while len(CACHE) > 3:
            CACHE.popitem(last=False)
    return token


def get_recording(token):
    with CACHE_LOCK:
        if token not in CACHE:
            raise ValueError("Load the signal again; it is no longer in the local cache")
        return CACHE[token]


def style_figure(figure, title):
    return figure.update_layout(template="plotly_dark", title=title,
                                margin=dict(l=45, r=20, t=50, b=40), height=380)


def number(label, identity, value, minimum, step=1):
    return dbc.Col([dbc.Label(label), dbc.Input(id=identity, type="number", value=value,
                                               min=minimum, step=step)], md=2)


app.layout = dbc.Container([
    dcc.Store(id="recording-token"),
    html.H2("EEG · Espacio de estados y anticipación de despertares", className="mt-4"),
    html.P("Exploración multicanal y revisión de experimentos. Las trayectorias no certifican un atractor; "
           "el proyecto aún requiere validación con sueño real."),
    dbc.Tabs([
        dbc.Tab([
            dcc.Upload(id="signal-upload", children=dbc.Button("Elegir señal", className="my-3"), multiple=False),
            dbc.Row([number("fs (Hz), CSV/NPY", "fs-input", None, 1),
                     dbc.Col([dbc.Label("Unidad, CSV/NPY"), dcc.Dropdown(id="unit-input", options=["V", "uV"])], md=2),
                     dbc.Col([dbc.Label("Columna de tiempo (s), opcional"), dbc.Input(id="time-column")], md=4),
                     dbc.Col(dbc.Button("Cargar señal", id="load-button", className="mt-4"), md=3)]),
            html.Div(id="load-status", className="my-3"),
            dbc.Label("Canales simultáneos (máximo 16 en el visor; CLI sin este límite)"),
            dcc.Dropdown(id="channels", multi=True),
            dbc.Row([number("Inicio (s)", "start", 20, 0), number("Fin (s)", "end", 80, 1),
                     number("Dimensión por canal", "dimension", 4, 2),
                     number("Retardo fijo (ms)", "tau", 50, 1),
                     number("Pasaaltas (Hz)", "low", .5, .01, .1),
                     number("Pasabajas (Hz)", "high", 35, 1)], className="mt-3"),
            dbc.Row([number("Theiler (s)", "theiler", 1, 0, .1),
                     dbc.Col(dbc.Button("Estimar τ del primer canal", id="auto-button", className="mt-4"), md=4),
                     dbc.Col(dbc.Button("Analizar", id="run-button", color="info", className="mt-4"), md=2)]),
            html.Div(id="auto-status", className="my-2"),
            html.Div(id="analysis-status", className="my-3"),
            dcc.Loading(dbc.Row([dbc.Col(dcc.Graph(id="time-figure"), md=6),
                                dbc.Col(dcc.Graph(id="phase-figure"), md=6)])),
            dbc.Row([dbc.Col(dcc.Graph(id="psd-figure"), md=6), dbc.Col(dcc.Graph(id="dimension-figure"), md=6)]),
            html.Pre(id="diagnostics", style={"whiteSpace": "pre-wrap"}),
        ], label="Explorar señal"),
        dbc.Tab([
            html.P("Entrena con python -m awakening train. Carga aquí *_predictions.csv o la salida de predict.", className="mt-3"),
            dcc.Upload(id="forecast-upload", children=dbc.Button("Cargar probabilidades CSV"), multiple=False),
            html.Div(id="forecast-status", className="my-3"),
            dcc.Dropdown(id="forecast-record", placeholder="Registro"),
            dcc.Store(id="forecast-data"),
            dcc.Graph(id="forecast-figure"),
            html.P("La etiqueta 0/1 indica un despertar en el intervalo futuro objetivo, no el estado actual. "
                   "Los CSV de alertas y report.json guardan umbral, falsas alarmas y anticipación. "
                   "La CLI predict funciona sin etiquetas."),
        ], label="Probabilidad de despertar"),
    ]),
], fluid=True)


@app.callback(Output("recording-token", "data"), Output("channels", "options"),
              Output("channels", "value"), Output("end", "value"), Output("load-status", "children"),
              Input("load-button", "n_clicks"), State("signal-upload", "contents"),
              State("signal-upload", "filename"), State("fs-input", "value"),
              State("unit-input", "value"), State("time-column", "value"), prevent_initial_call=True)
def on_upload(clicks, contents, filename, fs, unit, time_column):
    try:
        if not contents:
            raise ValueError("Selecciona un archivo")
        recording = ingest_file(base64.b64decode(contents.split(",", 1)[1]), filename, fs, unit, time_column or None)
        duration = len(recording.data)/recording.fs
        return (store_recording(recording), recording.channels, recording.channels[:2], min(80, duration),
                f"{filename}: {len(recording.channels)} canales, {recording.fs:g} Hz, {duration:g} s. Unidades internas: V.")
    except Exception as error:
        return no_update, no_update, no_update, no_update, f"Error: {error}"


def selected_prefix(token, channels, start, end, low, high):
    rec = get_recording(token)
    if not channels or len(channels) > 16:
        raise ValueError("Selecciona entre 1 y 16 canales")
    if start is None or end is None or not 0 <= start < end <= len(rec.data)/rec.fs:
        raise ValueError("Ventana temporal inválida")
    if end-start < 10:
        raise ValueError("Usa al menos 10 segundos")
    cfg = Config(low_hz=float(low), high_hz=float(high))
    columns = [rec.channels.index(c) for c in channels]
    filtered = filter_causal(rec.data[:int(end*rec.fs), columns], rec.fs, cfg)
    return filtered[int(start*rec.fs):], rec.fs


@app.callback(Output("tau", "value"), Output("auto-status", "children"),
              Input("auto-button", "n_clicks"), State("recording-token", "data"),
              State("channels", "value"), State("start", "value"), State("end", "value"),
              State("low", "value"), State("high", "value"), prevent_initial_call=True)
def auto_tau(clicks, token, channels, start, end, low, high):
    try:
        data, fs = selected_prefix(token, channels, start, end, low, high)
        tau, _ = ami_delay(data[:, 0], int(.5*fs))
        return tau/fs*1000, f"τ estimado por AMI: {tau} muestras del primer canal. No garantiza optimalidad."
    except Exception as error:
        return no_update, f"Error: {error}"


@app.callback(Output("time-figure", "figure"), Output("phase-figure", "figure"),
              Output("psd-figure", "figure"), Output("dimension-figure", "figure"),
              Output("diagnostics", "children"), Output("analysis-status", "children"),
              Input("run-button", "n_clicks"), State("recording-token", "data"),
              State("channels", "value"), State("start", "value"), State("end", "value"),
              State("dimension", "value"), State("tau", "value"), State("low", "value"),
              State("high", "value"), State("theiler", "value"), prevent_initial_call=True)
def run_analysis(clicks, token, channels, start, end, dimension, tau_ms, low, high, theiler):
    try:
        data, fs = selected_prefix(token, channels, start, end, low, high)
        if dimension is None or int(dimension) != dimension or not 2 <= dimension <= 12:
            raise ValueError("Dimensión entera entre 2 y 12")
        cfg = Config(window_s=end-start, step_s=min(10, end-start), dimension=int(dimension),
                     tau_s=float(tau_ms)/1000, low_hz=float(low), high_hz=float(high), theiler_s=float(theiler))
        if np.any(data.std(axis=0) < 1e-15):
            raise ValueError("Canal constante: no se puede estimar geometría")
        scaled = (data-data.mean(axis=0))/data.std(axis=0)
        tau = max(1, int(round(cfg.tau_s*fs)))
        E = delay_embedding(scaled, tau, cfg.dimension)
        ids = np.unique(np.linspace(0, len(E)-1, min(3000, len(E))).astype(int))
        centered = E[ids]-E[ids].mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt[:3].T
        projected = np.pad(projected, ((0, 0), (0, max(0, 3-projected.shape[1]))))
        color = start+(ids+(cfg.dimension-1)*tau)/fs
        phase = go.Figure(go.Scatter3d(x=projected[:, 0], y=projected[:, 1], z=projected[:, 2], mode="lines",
                                      line=dict(color=color, colorscale="Viridis", width=3,
                                                colorbar=dict(title="Tiempo (s)"))))
        style_figure(phase, f"Proyección PCA 3D de {E.shape[1]} coordenadas")
        phase.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
        time_fig, psd_fig = go.Figure(), go.Figure()
        step = max(1, len(data)//5000)
        for i, ch in enumerate(channels):
            time_fig.add_trace(go.Scatter(x=start+np.arange(len(data))[::step]/fs,
                                          y=data[::step, i]*1e6, name=ch, mode="lines"))
            freq, psd = signal.welch(data[:, i]*1e6, fs=fs, nperseg=min(len(data), int(4*fs)))
            psd_fig.add_trace(go.Scatter(x=freq, y=10*np.log10(np.maximum(psd, 1e-30)), name=ch))
        style_figure(time_fig, "Señal filtrada causalmente · µV").update_xaxes(title="Tiempo (s)")
        style_figure(psd_fig, "PSD · dB respecto a 1 µV²/Hz").update_xaxes(title="Hz", range=[0, min(fs/2, 60)])
        curve = correlation_curve(E, int(theiler*fs))
        dim_fig = go.Figure(go.Scatter(x=curve["r"], y=curve["local_slope"], mode="lines+markers"))
        style_figure(dim_fig, "Pendiente local de correlación · no es un D₂ certificado").update_xaxes(type="log", title="Radio normalizado")
        fnn = fnn_curve(data[:, 0], tau, max_dimension=min(8, cfg.dimension+2), theiler=int(theiler*fs))
        metrics = geometry(data, fs, cfg)
        details = json.dumps({"embedding_shape": list(E.shape), "tau_samples": tau, "tau_ms_actual": tau/fs*1000,
                              "geometry": metrics, "fnn_first_channel": fnn,
                              "correlation": curve["note"]}, indent=2)
        note = "Transitorio inicial: la ventana incluye los primeros 20 s. " if start < 20 else ""
        return time_fig, phase, psd_fig, dim_fig, details, note+"Análisis exploratorio. La PCA se ajusta solo para esta figura; no compara ejes entre ventanas."
    except Exception as error:
        return no_update, no_update, no_update, no_update, no_update, f"Error: {error}"


@app.callback(Output("forecast-data", "data"), Output("forecast-record", "options"),
              Output("forecast-record", "value"), Output("forecast-status", "children"),
              Input("forecast-upload", "contents"), prevent_initial_call=True)
def upload_forecast(contents):
    try:
        raw = base64.b64decode(contents.split(",", 1)[1])
        frame = pd.read_csv(io.BytesIO(raw))
        if not {"time_s", "probability"} <= set(frame):
            raise ValueError("Se requieren time_s y probability")
        if not np.isfinite(frame[["time_s", "probability"]]).all().all() or not frame.probability.between(0, 1).all():
            raise ValueError("Probabilidades/tiempos inválidos")
        if "recording" not in frame:
            frame["recording"] = "registro"
        names = frame.recording.astype(str).unique().tolist()
        frame["recording"] = frame.recording.astype(str)
        synthetic = ("synthetic_model" in frame and
                     frame.synthetic_model.astype(str).str.lower().isin(["true", "1"]).any())
        note = " MODELO SINTÉTICO: solo demostración de software." if synthetic else " Comprueba procedencia y modelo en report.json."
        return frame.to_dict("records"), names, names[0], f"{len(frame)} decisiones."+note
    except Exception as error:
        return no_update, no_update, no_update, f"Error: {error}"


@app.callback(Output("forecast-figure", "figure"), Input("forecast-record", "value"),
              State("forecast-data", "data"), prevent_initial_call=True)
def forecast_figure(record, rows):
    frame = pd.DataFrame(rows)
    frame = frame[frame.recording == record].sort_values("time_s")
    fig = go.Figure(go.Scatter(x=frame.time_s, y=frame.probability, name="Probabilidad"))
    if "label" in frame:
        fig.add_trace(go.Scatter(x=frame.time_s, y=frame.label, mode="lines", name="Evento futuro (etiqueta)", line=dict(dash="dot")))
    if "alert" in frame:
        selected = frame[frame.alert.astype(str).str.lower().isin(["true", "1"])]
        fig.add_trace(go.Scatter(x=selected.time_s, y=selected.probability, mode="markers", name="Alertas", marker=dict(size=10, symbol="x")))
    return style_figure(fig, "Probabilidad de despertar futuro").update_yaxes(range=[-.05, 1.05]).update_xaxes(title="Tiempo (s)")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
