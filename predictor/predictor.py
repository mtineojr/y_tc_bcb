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
        "## Proyección diaria por escenario",
        "",
        f"*Supuestos Δblue/día — optimista: 0.0 | base: {round(delta_blue['base'],3)} | pesimista: {round(delta_blue['pesimista'],3)}*",
        "",
    ]
    proys = {esc: proyectar(oficial_0, blue_0, tasa_cierre, db, max(HORIZONTES))
             for esc, db in delta_blue.items()}
    lineas.append("| Día | Fecha | Optimista | Base | Pesimista |")
    lineas.append("|---|---|---|---|---|")
    for d in range(1, max(HORIZONTES) + 1):
        fecha_obj = (fecha_base + timedelta(days=d)).date()
        opt  = proys["optimista"][d-1]["oficial_proj"]
        base = proys["base"][d-1]["oficial_proj"]
        pes  = proys["pesimista"][d-1]["oficial_proj"]
        lineas.append(f"| +{d} | {fecha_obj} | {opt:.2f} | **{base:.2f}** | {pes:.2f} |")
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

    proys = {esc: proyectar(oficial_0, blue_0, tasa_cierre, db, max(HORIZONTES))
             for esc, db in delta_blue.items()}
    build_dashboard(df, proys, fecha_base, oficial_0, blue_0, tasa_cierre, delta_blue, ts)

    print(f"✅ Predicción generada | base: {fecha_base.date()} | oficial: {oficial_0} | brecha: {round(blue_0-oficial_0,3)}")
    print(f"   Escenario base +3d: {proyectar(oficial_0, blue_0, tasa_cierre, delta_blue['base'], 3)[-1]['oficial_proj']}")




# =============================================================================
# DASHBOARD HTML (GitHub Pages) — docs/index.html
# =============================================================================

def _svg_fanchart(df_hist, proys, fecha_base, width=860, height=380):
    """Fan chart SVG: histórico del oficial + 3 escenarios proyectados."""
    hist = df_hist.tail(14)
    dias_h = list(range(-len(hist) + 1, 1))
    vals_h = list(hist["oficial"])

    n_proj = len(proys["base"])
    dias_p = list(range(1, n_proj + 1))
    esc_vals = {e: [f["oficial_proj"] for f in proys[e]] for e in proys}

    todos = vals_h + esc_vals["optimista"] + esc_vals["base"] + esc_vals["pesimista"]
    ymin, ymax = min(todos) * 0.995, max(todos) * 1.005
    xmin, xmax = dias_h[0], dias_p[-1]

    ML, MR, MT, MB = 52, 20, 16, 34
    def X(d): return ML + (d - xmin) / (xmax - xmin) * (width - ML - MR)
    def Y(v): return MT + (1 - (v - ymin) / (ymax - ymin)) * (height - MT - MB)

    def path(ds, vs):
        return "M " + " L ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(ds, vs))

    # banda entre optimista y pesimista
    banda = ("M " + " L ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(dias_p, esc_vals["pesimista"]))
             + " L " + " L ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(dias_p[::-1], esc_vals["optimista"][::-1]))
             + " Z")

    # gridlines
    grids = ""
    import numpy as _np
    for gv in _np.linspace(ymin, ymax, 5):
        grids += f'<line x1="{ML}" y1="{Y(gv):.1f}" x2="{width-MR}" y2="{Y(gv):.1f}" stroke="#E8DDD4" stroke-width="1"/>'
        grids += f'<text x="{ML-8}" y="{Y(gv)+4:.1f}" text-anchor="end" font-size="11" fill="#9A8B80">{gv:.2f}</text>'

    # eje x: etiquetas cada 5 días
    xlabels = ""
    for d in [dias_h[0], -7, 0, 5, 10, 15]:
        if xmin <= d <= xmax:
            lbl = "hoy" if d == 0 else (f"+{d}d" if d > 0 else f"{d}d")
            xlabels += f'<text x="{X(d):.1f}" y="{height-10}" text-anchor="middle" font-size="11" fill="#9A8B80">{lbl}</text>'

    hoy_x = X(0)
    v0 = vals_h[-1]

    return f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Proyección del TC oficial">
{grids}{xlabels}
<line x1="{hoy_x:.1f}" y1="{MT}" x2="{hoy_x:.1f}" y2="{height-MB}" stroke="#1C1F28" stroke-width="1" stroke-dasharray="4 4" opacity="0.35"/>
<path d="{banda}" fill="#F0B387" opacity="0.28"/>
<path d="{path(dias_h, vals_h)}" fill="none" stroke="#1C1F28" stroke-width="2.5"/>
<path d="{path(dias_p, esc_vals['optimista'])}" fill="none" stroke="#8FA98F" stroke-width="2" stroke-dasharray="6 4"/>
<path d="{path(dias_p, esc_vals['base'])}" fill="none" stroke="#C45033" stroke-width="2.8"/>
<path d="{path(dias_p, esc_vals['pesimista'])}" fill="none" stroke="#B3543D" stroke-width="2" stroke-dasharray="2 4"/>
<circle cx="{hoy_x:.1f}" cy="{Y(v0):.1f}" r="4.5" fill="#1C1F28"/>
<text x="{hoy_x+8:.1f}" y="{Y(v0)-8:.1f}" font-size="12" font-weight="600" fill="#1C1F28">{v0:.2f}</text>
</svg>"""


def build_dashboard(df, proys, fecha_base, oficial_0, blue_0, tasa_cierre, delta_blue, ts):
    brecha = blue_0 - oficial_0
    base7 = proys["base"][6]["oficial_proj"]
    var7 = (base7 / oficial_0 - 1) * 100

    filas_tabla = ""
    for d in range(len(proys["base"])):
        fecha_obj = (fecha_base + pd.Timedelta(days=d + 1)).date().strftime("%d %b")
        o = proys["optimista"][d]["oficial_proj"]
        b = proys["base"][d]["oficial_proj"]
        p = proys["pesimista"][d]["oficial_proj"]
        filas_tabla += f"<tr><td>+{d+1}</td><td>{fecha_obj}</td><td>{o:.2f}</td><td class='base'>{b:.2f}</td><td>{p:.2f}</td></tr>"

    svg = _svg_fanchart(df, proys, fecha_base)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Proyección TC Oficial — Yanbal Bolivia</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --terracota:#CD6340; --terracota-osc:#C45033; --naranja:#F26B43;
  --salmon:#F0B387; --crema:#FFF9F5; --crema2:#FDF4EF; --oscuro:#1C1F28;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--crema); color:var(--oscuro); font-family:'Inter',system-ui,sans-serif; line-height:1.5; }}
.wrap {{ max-width:920px; margin:0 auto; padding:48px 24px 40px; }}
.eyebrow {{ font-size:12px; letter-spacing:0.14em; text-transform:uppercase; color:var(--terracota-osc); font-weight:600; margin-bottom:14px; }}
h1 {{ font-family:'Fraunces',serif; font-weight:300; font-size:clamp(26px,4.2vw,40px); line-height:1.18; max-width:30ch; margin-bottom:8px; }}
h1 strong {{ font-weight:400; color:var(--terracota-osc); }}
.sub {{ color:#6E5F55; font-size:15px; margin-bottom:36px; max-width:64ch; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:36px; }}
.kpi {{ background:var(--crema2); border:1px solid #EFE2D8; border-radius:10px; padding:16px 18px; }}
.kpi .v {{ font-family:'Fraunces',serif; font-size:28px; font-weight:400; }}
.kpi .l {{ font-size:12px; color:#8A7A6E; margin-top:2px; }}
.kpi.hoy {{ background:var(--terracota); border-color:var(--terracota); }}
.kpi.hoy .v, .kpi.hoy .l {{ color:#fff; }}
.card {{ background:#fff; border:1px solid #EFE2D8; border-radius:12px; padding:24px; margin-bottom:28px; }}
.card h2 {{ font-family:'Fraunces',serif; font-weight:400; font-size:19px; margin-bottom:4px; }}
.card .note {{ font-size:12.5px; color:#8A7A6E; margin-bottom:14px; }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:12.5px; color:#5E5248; margin-top:10px; }}
.legend span::before {{ content:""; display:inline-block; width:18px; height:3px; margin-right:6px; vertical-align:middle; border-radius:2px; }}
.legend .l-hist::before {{ background:var(--oscuro); }}
.legend .l-opt::before  {{ background:#8FA98F; }}
.legend .l-base::before {{ background:var(--terracota-osc); }}
.legend .l-pes::before  {{ background:#B3543D; }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
th {{ text-align:left; font-size:11.5px; letter-spacing:0.08em; text-transform:uppercase; color:#8A7A6E; padding:8px 10px; border-bottom:2px solid var(--salmon); }}
td {{ padding:7px 10px; border-bottom:1px solid #F3E9E1; font-variant-numeric:tabular-nums; }}
td.base {{ font-weight:600; color:var(--terracota-osc); }}
tr:nth-child(even) td {{ background:var(--crema2); }}
.foot {{ font-size:12px; color:#9A8B80; border-top:1px solid #EFE2D8; padding-top:16px; }}
.foot .mark {{ font-family:'Fraunces',serif; letter-spacing:0.3em; text-align:center; margin-top:18px; color:var(--terracota); font-size:13px; }}
svg {{ width:100%; height:auto; display:block; }}
@media (max-width:560px) {{ .wrap {{ padding:32px 16px; }} .card {{ padding:16px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">🦋 Proyección de Tipo de Cambio · Análisis Comercial</div>
  <h1>El TC oficial cerraría en <strong>{base7:.2f} Bs/USD</strong> en una semana si el USDT mantiene su ritmo actual.</h1>
  <p class="sub">Escenario base: el BCB continúa cerrando ~{tasa_cierre*100:.0f}% de la brecha con el paralelo por día. Rango a 7 días: {proys["optimista"][6]["oficial_proj"]:.2f} (USDT se enfría) — {proys["pesimista"][6]["oficial_proj"]:.2f} (ráfaga continúa).</p>

  <div class="kpis">
    <div class="kpi hoy"><div class="v">{oficial_0:.2f}</div><div class="l">TC Oficial · {fecha_base.date().strftime("%d %b %Y")}</div></div>
    <div class="kpi"><div class="v">{blue_0:.2f}</div><div class="l">USDT paralelo (cierre)</div></div>
    <div class="kpi"><div class="v">{brecha:.2f}</div><div class="l">Brecha ({brecha/oficial_0*100:.1f}%)</div></div>
    <div class="kpi"><div class="v">{var7:+.1f}%</div><div class="l">Var. proyectada 7 días (base)</div></div>
  </div>

  <div class="card">
    <h2>Trayectoria proyectada a 15 días</h2>
    <div class="note">Histórico post-flotación + tres escenarios según comportamiento del USDT</div>
    {svg}
    <div class="legend">
      <span class="l-hist">Histórico oficial</span>
      <span class="l-opt">Optimista (USDT plano)</span>
      <span class="l-base">Base (mediana 7d: {delta_blue["base"]:+.3f}/día)</span>
      <span class="l-pes">Pesimista (P75: {delta_blue["pesimista"]:+.3f}/día)</span>
    </div>
  </div>

  <div class="card">
    <h2>Detalle diario</h2>
    <div class="note">Valores proyectados del TC oficial por escenario</div>
    <table>
      <thead><tr><th>Día</th><th>Fecha</th><th>Optimista</th><th>Base</th><th>Pesimista</th></tr></thead>
      <tbody>{filas_tabla}</tbody>
    </table>
  </div>

  <div class="foot">
    Modelo estructural brecha × tasa de cierre, calibrado con datos post-flotación (desde 27-jun-2026).
    El oficial replica al USDT con ~1 día de rezago (correlación 0.87). Proyecciones a +15 días son ilustrativas.
    Un cambio de política del BCB invalida el modelo. No constituye asesoría financiera.
    Corrida: {ts} UTC · Fuente: BCB (scraping diario) + USDT paralelo (carga manual).
    <div class="mark">Y A N B A L</div>
  </div>
</div>
</body>
</html>"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Dashboard generado: docs/index.html")


if __name__ == "__main__":
    main()
