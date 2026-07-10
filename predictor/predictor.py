"""
predictor.py — Predicción del TC Oficial BCB
Modelo estructural: brecha (blue - oficial) × tasa de cierre diaria.

Lee:
    data/tc_bcb.csv     (auto, scraper diario)
    data/usdt_blue.csv  (manual, intradía cada ~15 min)

Escribe:
    output/predicciones.csv
    output/reporte.md

Escenarios sobre el comportamiento del blue/USDT:
    optimista : blue se enfría (Δ = 0)
    base      : blue sigue la mediana de sus últimos 7 días
    pesimista : blue mantiene ritmo alto (percentil 75 de sus últimos 7 días)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

TC_CSV    = "data/tc_bcb.csv"
USDT_CSV  = "data/usdt_blue.csv"
OUT_CSV   = "output/predicciones.csv"
OUT_MD    = "output/reporte.md"

HORIZONTES = [3, 7, 15]
VENTANA_CIERRE = 5   # días para estimar la tasa de cierre del BCB
VENTANA_BLUE   = 7   # días para estimar el ritmo del blue


def cargar_datos():
    # --- TC oficial (ya viene 1 fila por día) ---
    tc = pd.read_csv(TC_CSV, parse_dates=["fecha_vigencia"])
    tc = tc[["fecha_vigencia", "tc_oficial"]].rename(
        columns={"fecha_vigencia": "fecha", "tc_oficial": "oficial"}
    )
    tc = tc.sort_values("fecha").reset_index(drop=True)

    # --- USDT/blue (intradía → agregar a diario) ---
    blue_raw = pd.read_csv(USDT_CSV, parse_dates=["datetime"])
    blue_raw["fecha"] = blue_raw["datetime"].dt.normalize()
    blue = blue_raw.groupby("fecha").agg(
        blue_cierre=("blue_sell", "last"),
        blue_promedio=("blue_sell", "mean"),
        n_obs=("blue_sell", "size"),
    ).reset_index()
    blue["blue_promedio"] = blue["blue_promedio"].round(4)

    df = pd.merge(tc, blue, on="fecha", how="inner").sort_values("fecha")
    df["brecha"] = df["blue_cierre"] - df["oficial"]
    df["d_oficial"] = df["oficial"].diff()
    df["d_blue"] = df["blue_cierre"].diff()
    # tasa de cierre: qué % de la brecha de ayer cerró el oficial hoy
    df["tasa_cierre"] = df["d_oficial"] / df["brecha"].shift(1)
    return df


def estimar_parametros(df):
    # Solo días hábiles con movimiento (el BCB no mueve en fin de semana)
    movs = df[df["d_oficial"] > 0].tail(VENTANA_CIERRE)
    tasa_cierre = movs["tasa_cierre"].median() if len(movs) else 0.0

    dblue = df["d_blue"].dropna().tail(VENTANA_BLUE)
    delta_blue = {
        "optimista": 0.0,
        "base":      float(dblue.median()),
        "pesimista": float(dblue.quantile(0.75)),
    }
    return float(tasa_cierre), delta_blue


def proyectar(oficial_0, blue_0, tasa_cierre, delta_blue, dias):
    """Proyección recursiva día a día."""
    filas = []
    oficial, blue = oficial_0, blue_0
    for d in range(1, dias + 1):
        blue = blue + delta_blue
        brecha = max(blue - oficial, 0)
        paso = brecha * tasa_cierre
        oficial = oficial + paso
        filas.append({"dia": d, "oficial_proj": round(oficial, 4),
                      "blue_proj": round(blue, 4), "brecha_proj": round(brecha, 4)})
    return filas


def main():
    df = cargar_datos()
    tasa_cierre, delta_blue = estimar_parametros(df)

    ult = df.iloc[-1]
    fecha_base = ult["fecha"]
    oficial_0, blue_0 = float(ult["oficial"]), float(ult["blue_cierre"])

    os.makedirs("output", exist_ok=True)

    # --- predicciones.csv ---
    registros = []
    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for esc, db in delta_blue.items():
        proy = proyectar(oficial_0, blue_0, tasa_cierre, db, max(HORIZONTES))
        for fila in proy:
            registros.append({
                "corrida_utc": ts,
                "fecha_base": fecha_base.date(),
                "escenario": esc,
                "dia_adelante": fila["dia"],
                "fecha_objetivo": (fecha_base + timedelta(days=fila["dia"])).date(),
                "oficial_proyectado": fila["oficial_proj"],
                "blue_proyectado": fila["blue_proj"],
                "tasa_cierre_usada": round(tasa_cierre, 4),
                "delta_blue_usado": round(db, 4),
            })
    pred = pd.DataFrame(registros)

    # Append al histórico de predicciones (trazabilidad de corridas)
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        pred = pd.concat([prev, pred], ignore_index=True)
    pred.to_csv(OUT_CSV, index=False)

    # --- reporte.md ---
    lineas = [
        f"# Reporte de predicción TC Oficial — {fecha_base.date()}",
        "",
        f"*Corrida: {ts} UTC*",
        "",
        "## Situación actual",
        "",
        f"| | Valor |",
        f"|---|---|",
        f"| TC Oficial | **{oficial_0}** |",
        f"| Blue/USDT (cierre) | {blue_0} |",
        f"| Brecha | {round(blue_0 - oficial_0, 3)} ({round((blue_0-oficial_0)/oficial_0*100,2)}%) |",
        f"| Tasa de cierre BCB (mediana {VENTANA_CIERRE} movs) | {round(tasa_cierre*100,1)}% de la brecha/día |",
        f"| Δ blue mediana {VENTANA_BLUE}d | {round(delta_blue['base'],4)} |",
        "",
        "## Proyecciones por escenario",
        "",
    ]
    for esc, db in delta_blue.items():
        proy = proyectar(oficial_0, blue_0, tasa_cierre, db, max(HORIZONTES))
        lineas.append(f"### {esc.capitalize()} (Δblue = {round(db,3)}/día)")
        lineas.append("")
        lineas.append("| Horizonte | Fecha | Oficial proyectado |")
        lineas.append("|---|---|---|")
        for h in HORIZONTES:
            fila = proy[h - 1]
            fecha_obj = (fecha_base + timedelta(days=h)).date()
            lineas.append(f"| +{h} días | {fecha_obj} | **{fila['oficial_proj']}** |")
        lineas.append("")
    lineas += [
        "---",
        "",
        "**Advertencias:** modelo estructural con historia corta (post-flotación). ",
        "Proyecciones a +15 días son ilustrativas, no confiables. ",
        "Supone que el BCB mantiene su patrón de cierre de brecha reciente; ",
        "un cambio de política (fijación, salto discreto, intervención) invalida el modelo. ",
        "No es asesoría financiera.",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    print(f"✅ Predicción generada | base: {fecha_base.date()} | oficial: {oficial_0} | brecha: {round(blue_0-oficial_0,3)}")
    print(f"   Escenario base +3d: {proyectar(oficial_0, blue_0, tasa_cierre, delta_blue['base'], 3)[-1]['oficial_proj']}")


if __name__ == "__main__":
    main()
