"""
bcb_scraper.py
Scraping del Tipo de Cambio Oficial del BCB.
Corre desde GitHub Actions — no requiere Databricks ni dependencias pesadas.
Selector confirmado: span.bcb-tco-num
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
import csv
import os

BCB_URL  = "https://www.bcb.gob.bo/"
CSV_PATH = "data/tc_bcb.csv"
HEADERS  = ["fecha_vigencia", "tc_oficial", "fecha_publicacion", "fuente"]


def scrape_tc() -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "es-BO,es;q=0.9",
    }

    resp = requests.get(BCB_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tc_span = soup.select_one("span.bcb-tco-num")
    if not tc_span:
        raise ValueError(
            "No se encontró span.bcb-tco-num. "
            "El BCB puede haber cambiado la estructura HTML."
        )

    tc_str    = tc_span.get_text(strip=True).replace(",", ".")
    tc_oficial = float(tc_str)

    tz_bo             = pytz.timezone("America/La_Paz")
    ahora             = datetime.now(tz_bo)
    fecha_vigencia    = ahora.strftime("%Y-%m-%d")
    fecha_publicacion = ahora.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "fecha_vigencia":    fecha_vigencia,
        "tc_oficial":        tc_oficial,
        "fecha_publicacion": fecha_publicacion,
        "fuente":            "BCB_scraping"
    }


def upsert_csv(nuevo: dict):
    """
    Lee el CSV existente, reemplaza la fila de esa fecha si ya existe
    (idempotente), agrega si es nueva, y reescribe el archivo.
    """
    filas = []

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for fila in reader:
                # Excluir la fila de la misma fecha para hacer upsert
                if fila["fecha_vigencia"] != nuevo["fecha_vigencia"]:
                    filas.append(fila)

    filas.append(nuevo)
    filas.sort(key=lambda x: x["fecha_vigencia"], reverse=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(filas)

    print(f"✅ CSV actualizado: {nuevo['fecha_vigencia']} = {nuevo['tc_oficial']} BOB/USD")


if __name__ == "__main__":
    datos = scrape_tc()
    print(f"TC capturado: {datos['tc_oficial']} | Fecha: {datos['fecha_vigencia']}")
    upsert_csv(datos)
