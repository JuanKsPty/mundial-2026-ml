"""Genera docs/pipeline.png: diagrama del pipeline del predictor."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#0F172A"
BODY = "#334155"
ARROW = "#94A3B8"

C_SRC = ("#DBEAFE", "#2563EB")   # fuentes de datos
C_PREP = ("#FEF3C7", "#B45309")  # datos / features
C_MODEL = ("#DCFCE7", "#15803D") # modelos
C_PRED = ("#EDE9FE", "#6D28D9")  # predicción
C_OUT = ("#F1F5F9", "#475569")   # salidas / dashboard

fig, ax = plt.subplots(figsize=(13.5, 10.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, body, colors, title_size=11.5, body_size=9.5):
    face, edge = colors
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=face, edgecolor=edge, linewidth=1.6, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h / 2 - 2.2, body, ha="center", va="center",
            fontsize=body_size, color=BODY, zorder=3, linespacing=1.45)


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=16,
            linewidth=1.8, color=ARROW, zorder=1,
            shrinkA=2, shrinkB=2,
        )
    )


# ---- Fila 1: fuentes de datos -------------------------------------------
box(3, 86, 28, 11, "Histórico internacional",
    "martj42 (CSV, GitHub)\n~25,400 partidos desde 2000", C_SRC)
box(36, 86, 28, 11, "Ranking FIFA",
    "ranking oficial por fecha\n(fuerza relativa de selecciones)", C_SRC)
box(69, 86, 28, 11, "Mundial 2026 en curso",
    "API-Football → martj42 → CSV manual\n(cascada de fallbacks) → SQLite", C_SRC)

# ---- Fila 2: preparación de datos ---------------------------------------
box(20, 71, 60, 10, "1 · Datos  —  pandas + sqlite3",
    "limpieza, normalización de nombres y unión de fuentes\nen un solo dataset de partidos", C_PREP)

# ---- Fila 3: features -----------------------------------------------------
box(20, 56, 60, 10, "2 · Features walk-forward  —  pandas / numpy",
    "Elo, forma reciente, head-to-head, ranking, logros — cada partido\nusa solo información anterior a su fecha (sin fuga de datos)", C_PREP)

# ---- Fila 4: modelos -------------------------------------------------------
box(3, 37, 45, 13, "3a · Clasificador W / D / L  —  scikit-learn",
    "GradientBoostingClassifier + CalibratedClassifierCV\n(sigmoid) · split temporal: train ≤2021 · cal. 2022-24\n· test 2025+  →  P(victoria / empate / derrota)", C_MODEL)
box(52, 37, 45, 13, "3b · Modelo de goles  —  scikit-learn",
    "2 × PoissonRegressor (goles esperados λ por equipo)\n+ ajuste Dixon-Coles (ρ)\n→  matriz 7×7 de probabilidad de marcadores", C_MODEL)

# ---- Fila 5: predicción ----------------------------------------------------
box(25, 22, 50, 10, "4 · predict(equipo1, equipo2)",
    "resultado + marcador exacto (condicionado al resultado)\n+ probabilidades calibradas", C_PRED)

# ---- Fila 6: salidas -------------------------------------------------------
box(3, 4, 45, 12, "5 · Simulación Monte Carlo  —  numpy",
    "miles de torneos: resultado ~ multinomial ·\nmarcador ~ matriz Poisson\n→ P(campeón) y avance por ronda", C_OUT)
box(52, 4, 45, 12, "6 · Dashboard  —  Streamlit + plotly",
    "predicción con análisis, bracket interactivo,\nsimulación y estado real del torneo", C_OUT)

# ---- Flechas ---------------------------------------------------------------
arrow(17, 85.5, 40, 81.8)    # martj42 -> datos
arrow(50, 85.5, 50, 81.8)    # ranking -> datos
arrow(83, 85.5, 60, 81.8)    # wc2026 -> datos
arrow(50, 70.5, 50, 66.8)    # datos -> features
arrow(40, 55.5, 28, 50.8)    # features -> clasificador
arrow(60, 55.5, 72, 50.8)    # features -> poisson
arrow(28, 36.5, 40, 32.8)    # clasificador -> predict
arrow(72, 36.5, 60, 32.8)    # poisson -> predict
arrow(40, 21.5, 28, 16.8)    # predict -> monte carlo
arrow(60, 21.5, 72, 16.8)    # predict -> dashboard
arrow(48.5, 10, 51.5, 10)    # monte carlo -> dashboard

fig.savefig("/Users/juank/dev/mundial-2026-ml/docs/pipeline.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
