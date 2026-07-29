"""
predictor.py — v2 (fuente: API dolarbluebolivia Backend V3)
Predicción del TC oficial mediante modelo estructural brecha × tasa de cierre.

Cambios vs v1:
  - Fuente USDT: API (antes CSV manual)
  - Oficial para la brecha: official.sell de la API (misma fuente que el blue)
  - Escenarios: percentiles REALES de Δblue desde rolling-returns (antes ad-hoc)
  - Validación cruzada: compara oficial_api vs scraper BCB (data/tc_bcb.csv)

Salidas: output/predicciones.csv, output/reporte.md,
         docs/index.html, docs/metodologia.html
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone

# Asegurar que los módulos hermanos (data_source, dashboards) sean importables
# sin importar desde qué directorio se invoque el script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_source as ds

TC_BCB_CSV = "data/tc_bcb.csv"
OUT_CSV    = "output/predicciones.csv"
OUT_MD     = "output/reporte.md"
HORIZONTES = [3, 7, 15]
VENTANA_CIERRE = 5
UMBRAL_DIVERGENCIA = 0.05   # alerta si |oficial_api - bcb| supera esto


# -------------------------------------------------------------------------
# Construcción del dataset diario + tasa de cierre
# -------------------------------------------------------------------------
def construir_dataset():
    diario = ds.get_historico_diario()
    diario = diario.sort_values("fecha").reset_index(drop=True)

    diario["brecha"]      = diario["blue_cierre"] - diario["oficial_api"]
    diario["d_oficial"]   = diario["oficial_api"].diff()
    diario["tasa_cierre"] = diario["d_oficial"] / diario["brecha"].shift(1)
    return diario


def validar_cruzado(diario):
    """Compara el oficial de la API contra el scraper BCB. Devuelve texto de alerta."""
    if not os.path.exists(TC_BCB_CSV):
        return "Scraper BCB no disponible para validación cruzada."
    bcb = pd.read_csv(TC_BCB_CSV, parse_dates=["fecha_vigencia"])
    bcb = bcb.rename(columns={"fecha_vigencia": "fecha", "tc_oficial": "oficial_bcb"})
    merge = pd.merge(
        diario[["fecha", "oficial_api"]],
        bcb[["fecha", "oficial_bcb"]],
        on="fecha", how="inner"
    )
    if merge.empty:
        return "Sin fechas coincidentes entre API y BCB para validar."
    merge["dif"] = (merge["oficial_api"] - merge["oficial_bcb"]).abs()
    max_dif = merge["dif"].max()
    n_div = int((merge["dif"] > UMBRAL_DIVERGENCIA).sum())
    if n_div == 0:
        return f"✅ API y BCB coinciden ({len(merge)} días comparados, dif. máx {max_dif:.3f})."
    return (f"⚠️ {n_div} de {len(merge)} días con divergencia > {UMBRAL_DIVERGENCIA} "
            f"entre API y BCB (máx {max_dif:.3f}). Revisar fuentes.")


def estimar_tasa_cierre(diario):
    movs = diario[diario["d_oficial"] > 0].tail(VENTANA_CIERRE)
    return float(movs["tasa_cierre"].median()) if len(movs) else 0.0


# -------------------------------------------------------------------------
# Proyección recursiva
# -------------------------------------------------------------------------
def proyectar(oficial_0, blue_0, tasa_cierre, delta_blue, dias):
    filas = []
    oficial, blue = oficial_0, blue_0
    for d in range(1, dias + 1):
        blue  += delta_blue
        brecha = max(blue - oficial, 0)
        oficial += brecha * tasa_cierre
        filas.append({"dia": d, "oficial_proj": round(oficial, 4),
                      "blue_proj": round(blue, 4), "brecha_proj": round(brecha, 4)})
    return filas


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    diario      = construir_dataset()
    tasa_cierre = estimar_tasa_cierre(diario)
    deltas      = ds.get_delta_blue_percentiles()
    delta_blue  = {k: deltas[k] for k in ("optimista", "base", "pesimista")}
    alerta_val  = validar_cruzado(diario)
    fresh       = ds.get_freshness()

    ult = diario.iloc[-1]
    fecha_base = ult["fecha"]
    oficial_0  = float(ult["oficial_api"])
    blue_0     = float(ult["blue_cierre"])

    proys = {e: proyectar(oficial_0, blue_0, tasa_cierre, db, max(HORIZONTES))
             for e, db in delta_blue.items()}

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("output", exist_ok=True)

    # --- predicciones.csv (append histórico de corridas) ---
    registros = []
    for esc, db in delta_blue.items():
        for fila in proys[esc]:
            registros.append({
                "corrida_utc": ts, "fecha_base": fecha_base.date(), "escenario": esc,
                "dia_adelante": fila["dia"],
                "fecha_objetivo": (fecha_base + timedelta(days=fila["dia"])).date(),
                "oficial_proyectado": fila["oficial_proj"],
                "blue_proyectado": fila["blue_proj"],
                "tasa_cierre_usada": round(tasa_cierre, 4),
                "delta_blue_usado": round(db, 4),
            })
    pred = pd.DataFrame(registros)
    if os.path.exists(OUT_CSV):
        pred = pd.concat([pd.read_csv(OUT_CSV), pred], ignore_index=True)
    pred.to_csv(OUT_CSV, index=False)

    # --- reporte.md ---
    escribir_reporte(diario, proys, fecha_base, oficial_0, blue_0,
                     tasa_cierre, delta_blue, deltas, alerta_val, fresh, ts)

    # --- dashboards HTML ---
    import dashboards
    dashboards.build_dashboard(diario, proys, fecha_base, oficial_0, blue_0,
                               tasa_cierre, delta_blue, deltas, ts)
    dashboards.build_metodologia(diario, deltas, tasa_cierre, ts)

    print(f"✅ Predicción v2 | base {fecha_base.date()} | oficial {oficial_0} | "
          f"blue {blue_0} | brecha {round(blue_0-oficial_0,3)}")
    print(f"   {alerta_val}")
    print(f"   Base +7d: {proys['base'][6]['oficial_proj']}")


def escribir_reporte(diario, proys, fecha_base, oficial_0, blue_0,
                     tasa_cierre, delta_blue, deltas, alerta_val, fresh, ts):
    brecha = blue_0 - oficial_0
    L = [
        f"# Reporte de predicción TC Oficial — {fecha_base.date()}",
        "", f"*Corrida: {ts} UTC · Fuente: API dolarbluebolivia (Backend V3)*", "",
        f"**Validación cruzada:** {alerta_val}", "",
        "## Situación actual", "",
        "| Indicador | Valor |", "|---|---|",
        f"| TC Oficial (API) | **{oficial_0:.2f}** |",
        f"| Blue/USDT (cierre) | {blue_0:.2f} |",
        f"| Brecha | {brecha:.2f} ({brecha/oficial_0*100:.2f}%) |",
        f"| Tasa de cierre BCB (mediana {VENTANA_CIERRE}) | {tasa_cierre*100:.1f}%/día |",
        f"| Volatilidad diaria Δblue | {deltas['vol_diaria']:.3f} Bs |",
        f"| Muestras históricas | {deltas['n_muestras']} |",
        "", "## Proyección diaria por escenario", "",
        f"*Δblue/día — optimista: 0.0 | base: {delta_blue['base']:+.3f} | pesimista: {delta_blue['pesimista']:+.3f}*",
        "", "| Día | Fecha | Optimista | Base | Pesimista |", "|---|---|---|---|---|",
    ]
    for d in range(max(HORIZONTES)):
        f_obj = (fecha_base + timedelta(days=d+1)).date()
        o = proys["optimista"][d]["oficial_proj"]
        b = proys["base"][d]["oficial_proj"]
        p = proys["pesimista"][d]["oficial_proj"]
        L.append(f"| +{d+1} | {f_obj} | {o:.2f} | **{b:.2f}** | {p:.2f} |")
    L += ["", "---", "",
          "**Advertencias:** modelo estructural, historia post-flotación corta. ",
          "Proyecciones a +15 días ilustrativas. Un cambio de política del BCB lo invalida. ",
          "No es asesoría financiera."]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
