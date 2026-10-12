import base64
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from awakening.io import Recording

pytest.importorskip("dash")
spec = importlib.util.spec_from_file_location("dashboard", Path(__file__).parents[1]/"atractor.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def test_dash_layout_and_dependencies():
    client = dashboard.app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/_dash-dependencies").status_code == 200


def test_upload_error_returns_all_outputs():
    result = dashboard.on_upload(1, None, None, None, None, None)
    assert len(result) == 5
    assert "Error" in result[-1]


def test_multichannel_dimension_is_preserved():
    t = np.arange(12000)/100
    data = np.column_stack([np.sin(2*np.pi*3*t), np.cos(2*np.pi*7*t)])*1e-5
    token = dashboard.store_recording(Recording(data, 100, ["a", "b"]))
    result = dashboard.run_analysis(1, token, ["a", "b"], 20, 80, 5, 50, .5, 35, 1)
    assert not result[-1].startswith("Error"), result[-1]
    assert json.loads(result[-2])["embedding_shape"][1] == 10
    assert len(result[0].data) == 2


def test_forecast_viewer_handles_predict_csv():
    text = b"time_s,probability,alert\n80,0.2,False\n90,0.8,True\n"
    rows, _, selected, status = dashboard.upload_forecast("data:text/csv;base64,"+base64.b64encode(text).decode())
    assert not status.startswith("Error")
    assert len(dashboard.forecast_figure(selected, rows).data) == 2
