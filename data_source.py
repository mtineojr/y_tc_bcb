"""
data_source.py — Capa de datos del predictor TC (Backend V3 dolarbluebolivia)
Encapsula todas las llamadas a la API y la agregación diaria.

La API key se lee de la variable de entorno DOLARBLUE_API_KEY
(inyectada por GitHub Actions desde el secret del repo).
"""

import os
import time
import requests
import pandas as pd

API_BASE = "https://api.dolarbluebolivia.click"
API_KEY  = os.environ.get("DOLARBLUE_API_KEY", "")

# Rutas
EP_OFFICIAL_RATE = f"{API_BASE}/v1/officialRate"                       # público
EP_EXPORT        = f"{API_BASE}/private/v1/history/export"             # privado
EP_ROLLING       = f"{API_BASE}/private/v1/history/rolling-returns"    # privado
EP_MOVING_AVG    = f"{API_BASE}/private/v1/history/moving-average"     # privado
EP_FRESHNESS     = f"{API_BASE}/private/v1/system/freshness"           # privado

VENTANA_DIAS = 90


def _headers():
    if not API_KEY:
        raise RuntimeError(
            "Falta DOLARBLUE_API_KEY. En GitHub Actions se inyecta desde el "
            "secret del repo; en local, exportar la variable de entorno."
        )
    return {"X-API-Key": API_KEY}


def _rango_ts(dias=VENTANA_DIAS):
    to_ts   = int(time.time())
    from_ts = to_ts - dias * 24 * 3600
    return {"from_ts": from_ts, "to_ts": to_ts}


# -------------------------------------------------------------------------
# 1. Snapshot actual (público) — blue + oficial en tiempo real
# -------------------------------------------------------------------------
def get_snapshot() -> dict:
    r = requests.get(EP_OFFICIAL_RATE, timeout=10)
    r.raise_for_status()
    d = r.json()["data"]
    return {
        "blue_sell":     d["blue"]["sell"],
        "blue_buy":      d["blue"]["buy"],
        "official_sell": d["official"]["sell"],
        "official_buy":  d["official"]["buy"],
        "fetched_at":    d["fetched_at"],
    }


# -------------------------------------------------------------------------
# 2. Histórico intradía → agregación diaria (blue + oficial de la API)
# -------------------------------------------------------------------------
def get_historico_diario(dias=VENTANA_DIAS) -> pd.DataFrame:
    r = requests.get(EP_EXPORT, headers=_headers(),
                     params={**_rango_ts(dias), "format": "json"}, timeout=30)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())

    raw["fecha"] = pd.to_datetime(raw["timestamp"], unit="s").dt.normalize()
    diario = raw.groupby("fecha").agg(
        blue_cierre    =("blue_sell",     "last"),
        blue_promedio  =("blue_sell",     "mean"),
        oficial_api    =("official_sell", "last"),
        n_obs          =("blue_sell",     "size"),
    ).reset_index()
    diario["blue_promedio"] = diario["blue_promedio"].round(4)
    return diario


# -------------------------------------------------------------------------
# 3. Rolling returns → percentiles REALES de Δblue diario (para escenarios)
# -------------------------------------------------------------------------
def get_delta_blue_percentiles(dias=VENTANA_DIAS, ventana_reciente=7) -> dict:
    """
    Descarga rolling-returns y calcula percentiles del cambio diario absoluto
    del blue. Usa la ventana reciente para reflejar el régimen actual, no toda
    la historia (que incluye la era del tipo fijo).
    """
    r = requests.get(EP_ROLLING, headers=_headers(),
                     params={**_rango_ts(dias), "period": "daily"}, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["fecha"]      = pd.to_datetime(df["t"], unit="s").dt.normalize()
    df["d_blue_abs"] = df["close"] - df["prev_close"]

    reciente = df.tail(ventana_reciente)["d_blue_abs"]
    completo = df["d_blue_abs"]

    return {
        "optimista": 0.0,                                  # blue plano
        "base":      float(reciente.median()),             # mediana reciente
        "pesimista": float(reciente.quantile(0.75)),       # P75 reciente
        # métricas de contexto
        "vol_diaria":       float(completo.std()),
        "p90_delta":        float(completo.quantile(0.90)),
        "n_muestras":       int(len(df)),
        "serie_close":      df[["fecha", "close"]].to_dict("records"),
    }


# -------------------------------------------------------------------------
# 4. Freshness — para estampar el reporte con el lag de datos
# -------------------------------------------------------------------------
def get_freshness() -> dict:
    try:
        r = requests.get(EP_FRESHNESS, headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}
