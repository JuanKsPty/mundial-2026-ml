# UI text is in Spanish (academic deliverable); code stays in English.

"""Dashboard: match predictor, tournament simulation, WC2026 state, methodology.

Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow `from src...`

from src.config import (
    FIGURES_DIR,
    MAX_GOALS,
    N_SIMULATIONS_DEFAULT,
    OUTPUTS_DIR,
    WC2026_ALL_TEAMS,
)
from src.data import db
from src.utils import viz

st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")

PROB_LABELS = {"home": "Victoria", "draw": "Empate", "away": "Victoria"}


@st.cache_resource(show_spinner="Cargando modelos...")
def _load_predictor():
    from src.predict import predict_match
    return predict_match


@st.cache_resource(show_spinner="Preparando simulador...")
def _load_sampler():
    from src.simulation.montecarlo import make_sampler
    return make_sampler()


@st.cache_data(ttl=600)
def _load_matches():
    return db.load_matches(), db.last_sync()


def _prob_bar_chart(pred) -> go.Figure:
    p = pred.probabilities
    labels = [
        f"Victoria {pred.home_team}",
        "Empate",
        f"Victoria {pred.away_team}",
    ]
    values = [p["home"], p["draw"], p["away"]]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=[viz.BLUE, viz.INK_MUTED, viz.AQUA],
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=180, margin=dict(l=0, r=30, t=10, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=viz.GRID),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor=viz.SURFACE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=viz.INK_SECONDARY),
    )
    return fig


def _score_heatmap(pred) -> go.Figure:
    m = pred.score_matrix
    goals = list(range(MAX_GOALS + 1))
    fig = go.Figure(go.Heatmap(
        z=m * 100,
        x=[str(g) for g in goals], y=[str(g) for g in goals],
        colorscale=[[i / (len(viz.SEQ_BLUES) - 1), c] for i, c in enumerate(viz.SEQ_BLUES)],
        colorbar=dict(title="%", ticksuffix="%"),
        hovertemplate=(f"{pred.home_team} %{{y}} - %{{x}} {pred.away_team}"
                       "<br>prob: %{z:.1f}%<extra></extra>"),
    ))
    bh, ba = pred.predicted_score
    fig.add_shape(
        type="rect", x0=ba - 0.5, x1=ba + 0.5, y0=bh - 0.5, y1=bh + 0.5,
        line=dict(color=viz.INK, width=2),
    )
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title=f"Goles {pred.away_team}", yaxis_title=f"Goles {pred.home_team}",
        plot_bgcolor=viz.SURFACE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=viz.INK_SECONDARY),
    )
    return fig


def _champion_bar_chart(df: pd.DataFrame, prob_col: str, top_n: int = 12) -> go.Figure:
    top = df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top[prob_col], y=top["team"], orientation="h",
        marker_color=viz.BLUE,
        text=[f"{v:.1%}" for v in top[prob_col]],
        textposition="outside",
    ))
    fig.update_layout(
        height=30 * top_n + 60, margin=dict(l=0, r=40, t=10, b=10),
        xaxis=dict(tickformat=".0%", gridcolor=viz.GRID),
        plot_bgcolor=viz.SURFACE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=viz.INK_SECONDARY),
    )
    return fig


st.title("⚽ Predictor Mundial 2026")
st.caption(
    "Proyecto académico UTP-FISC · calibración temporal de probabilidades, "
    "Poisson/Dixon-Coles, backtest walk-forward, integración API-Football/SQLite "
    "y simulación desde el estado real del torneo."
)

tab_pred, tab_bracket, tab_sim, tab_estado, tab_metodo = st.tabs([
    "🔮 Predicción de partido", "🗂 Bracket", "🏆 Simulación del torneo",
    "📋 Estado del Mundial", "📖 Metodología",
])

ROUND_ES = {
    "round_of_16": "Octavos", "quarterfinal": "Cuartos",
    "semifinal": "Semifinal", "final": "Final",
}
# column layout of the real bracket: slots listed top-to-bottom so that
# adjacent pairs feed the slot to their right
BRACKET_COLUMNS = [
    ("Octavos", ["r16_90", "r16_89", "r16_93", "r16_94",
                 "r16_91", "r16_92", "r16_95", "r16_96"]),
    ("Cuartos", ["qf_97", "qf_98", "qf_99", "qf_100"]),
    ("Semifinales", ["sf_101", "sf_102"]),
    ("Final", ["final"]),
]


def _render_bracket_slot(slot_df: pd.DataFrame) -> None:
    """One knockout slot as a bordered card. slot_df: rows of
    bracket_probabilities.csv for a single slot."""
    first = slot_df.iloc[0]
    with st.container(border=True):
        st.caption(f"{ROUND_ES[first['round']]} · {first['date']}")
        if first["played"]:
            home = slot_df[slot_df.side == "home"].iloc[0]["team"]
            away = slot_df[slot_df.side == "away"].iloc[0]["team"]
            hg, ag = first["score"].split("-")
            winner = first["winner"]
            h_mark = " ✅" if winner == home else ""
            a_mark = " ✅" if winner == away else ""
            st.markdown(f"**{home}**{h_mark} &nbsp; {hg} — {ag} &nbsp; **{away}**{a_mark}")
        else:
            for side in ("home", "away"):
                cands = slot_df[slot_df.side == side].nlargest(2, "p_present")
                if len(cands) == 1:
                    r = cands.iloc[0]
                    st.markdown(f"**{r['team']}** · gana {r['p_win_match']:.0%}")
                else:
                    st.markdown(" · ".join(
                        f"{r.team} {r.p_present:.0%}" for r in cands.itertuples(index=False)
                    ))
            fav = slot_df.nlargest(1, "p_win_match").iloc[0]
            st.caption(f"Favorito: {fav['team']} ({fav['p_win_match']:.0%} avanza)")

# ---------------------------------------------------------------- tab 1
with tab_pred:
    col1, col2 = st.columns(2)
    home = col1.selectbox("Equipo 1", WC2026_ALL_TEAMS, index=WC2026_ALL_TEAMS.index("Argentina"))
    away = col2.selectbox("Equipo 2", WC2026_ALL_TEAMS, index=WC2026_ALL_TEAMS.index("France"))

    if home == away:
        st.warning("Elige dos equipos distintos.")
    else:
        pred = _load_predictor()(home, away)
        st.subheader("Probabilidades calibradas")
        st.plotly_chart(_prob_bar_chart(pred), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        result_txt = {"H": f"Gana {pred.home_team}", "D": "Empate", "A": f"Gana {pred.away_team}"}
        c1.metric("Resultado más probable", result_txt[pred.predicted_result])
        c2.metric("Marcador predicho", f"{pred.predicted_score[0]} - {pred.predicted_score[1]}")
        c3.metric(
            "Goles esperados (λ)",
            f"{pred.expected_goals[0]:.2f} - {pred.expected_goals[1]:.2f}",
        )

        col_hm, col_top = st.columns([3, 1])
        with col_hm:
            st.subheader("Matriz de marcadores (Poisson + Dixon-Coles)")
            st.plotly_chart(_score_heatmap(pred), use_container_width=True)
        with col_top:
            st.subheader("Top 5 marcadores")
            for (h, a), prob in pred.top_scores:
                st.write(f"**{h} - {a}** · {prob:.1%}")

        # ------------------------------------------- match analysis context
        st.divider()
        st.subheader("📊 Análisis del enfrentamiento")
        from src.features.live_features import get_live_state, to_wc_team

        state = get_live_state()
        ta, tb = to_wc_team(home), to_wc_team(away)
        comp = pd.DataFrame({
            "": ["Elo", "Puntos FIFA", "Forma (win rate últ. 5)",
                 "Victorias en tandas de penales"],
            ta: [f"{state.elo.get(ta, 1500):.0f}",
                 f"{state.fifa_points.get(ta, 1350):.0f}",
                 f"{state.last5_form.get(ta, 0.5):.0%}",
                 f"{state.penalty_win_rate.get(ta, 0.5):.0%}"],
            tb: [f"{state.elo.get(tb, 1500):.0f}",
                 f"{state.fifa_points.get(tb, 1350):.0f}",
                 f"{state.last5_form.get(tb, 0.5):.0%}",
                 f"{state.penalty_win_rate.get(tb, 0.5):.0%}"],
        })
        st.table(comp.set_index(""))

        h2h = state.h2h_stats(ta, tb)
        n_h2h = int(h2h["h2h_matches_played"])
        col_a, col_h2h, col_b = st.columns(3)
        with col_a:
            st.markdown(f"**Últimos partidos · {ta}**")
            st.dataframe(state.recent_matches(ta), use_container_width=True,
                         hide_index=True, height=215)
        with col_h2h:
            st.markdown("**Historial directo**")
            if n_h2h == 0:
                st.info("Nunca se han enfrentado en el histórico (desde 1872).")
            else:
                wins_a = round(h2h["h2h_home_win_rate"] * n_h2h)
                draws = round(h2h["h2h_draw_rate"] * n_h2h)
                st.markdown(
                    f"{n_h2h} partidos · **{ta} {wins_a}** · "
                    f"empates {draws} · **{tb} {n_h2h - wins_a - draws}**"
                )
                st.dataframe(state.h2h_matches(ta, tb), use_container_width=True,
                             hide_index=True, height=180)
        with col_b:
            st.markdown(f"**Últimos partidos · {tb}**")
            st.dataframe(state.recent_matches(tb), use_container_width=True,
                         hide_index=True, height=215)

        with st.expander("🔍 Features que ve el modelo (transparencia)"):
            feats = state.build_features(ta, tb)
            st.dataframe(
                pd.DataFrame({"feature": list(feats), "valor": list(feats.values())}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "G = ganó, E = empató, P = perdió. La predicción promedia ambas "
                "orientaciones del enfrentamiento (cancha neutral)."
            )

# ---------------------------------------------------------------- tab bracket
with tab_bracket:
    top_l, top_r = st.columns([3, 1])
    top_l.markdown(
        "Cuadro eliminatorio **real** del Mundial 2026. Los partidos jugados quedan "
        "fijos; en los pendientes se muestra la probabilidad de cada equipo de "
        "ocupar y ganar la llave (Monte Carlo, 5000 simulaciones)."
    )
    if top_r.button("🔄 Actualizar y re-simular", use_container_width=True):
        with st.spinner("Sincronizando resultados y re-simulando el bracket..."):
            from scripts.sync_wc2026 import main as _sync_main
            from src.simulation.tournament_state import simulate_from_current_state
            _sync_main(source="auto")
            simulate_from_current_state(N_SIMULATIONS_DEFAULT, sampler=_load_sampler())
        st.cache_data.clear()
        st.rerun()

    bracket_csv = OUTPUTS_DIR / "bracket_probabilities.csv"
    if not bracket_csv.exists():
        st.info("Aún no hay simulación del bracket - pulsa **Actualizar y re-simular**.")
    else:
        bp = pd.read_csv(bracket_csv)
        champ_row = bp[(bp["slot"] == "final")].nlargest(1, "p_win_match").iloc[0]
        st.caption(
            f"Favorito al título: **{champ_row['team']}** "
            f"({champ_row['p_win_match']:.0%}) · "
            f"actualizado con {int(bp[bp.played].slot.nunique())} llaves ya jugadas"
        )
        cols = st.columns(len(BRACKET_COLUMNS), gap="medium")
        for col, (title, slot_ids) in zip(cols, BRACKET_COLUMNS):
            with col:
                st.subheader(title)
                for sid in slot_ids:
                    slot_df = bp[bp["slot"] == sid]
                    if not slot_df.empty:
                        _render_bracket_slot(slot_df)

# ---------------------------------------------------------------- tab 2
with tab_sim:
    mode = st.radio(
        "Modo de simulación",
        ["Desde el estado actual del torneo", "Torneo completo desde fase de grupos"],
        horizontal=True,
    )
    n_sims = st.select_slider(
        "Número de simulaciones", options=[500, 1000, 2000, 5000, 10000],
        value=N_SIMULATIONS_DEFAULT,
    )

    current_mode = mode.startswith("Desde")
    csv_path = OUTPUTS_DIR / ("simulation_current.csv" if current_mode else "simulation_full.csv")

    if st.button("▶ Simular ahora", type="primary"):
        with st.spinner(f"Corriendo {n_sims} simulaciones..."):
            if current_mode:
                from src.simulation.tournament_state import simulate_from_current_state
                df_sim = simulate_from_current_state(n_sims, sampler=_load_sampler())
            else:
                from src.simulation.montecarlo import run_full_simulation
                df_sim = run_full_simulation(n_sims, n_workers=1)
    elif csv_path.exists():
        df_sim = pd.read_csv(csv_path)
        st.caption(f"Mostrando resultados precomputados ({csv_path.name}).")
    else:
        df_sim = None
        st.info("No hay resultados precomputados; pulsa **Simular ahora**.")

    if df_sim is not None:
        st.subheader("Probabilidad de ser campeón")
        st.plotly_chart(_champion_bar_chart(df_sim, "p_champion"), use_container_width=True)
        st.subheader("Detalle por equipo")
        pct_cols = [c for c in df_sim.columns if c.startswith("p_")]
        st.dataframe(
            df_sim.style.format({c: "{:.1%}" for c in pct_cols}
                                | {"expected_points": "{:.2f}"}),
            use_container_width=True, height=420,
        )

    if current_mode:
        try:
            from src.simulation.tournament_state import pending_bracket_summary
            st.subheader("Bracket real (jugados y pendientes)")
            st.dataframe(pending_bracket_summary(), use_container_width=True)
        except FileNotFoundError as e:
            st.warning(f"Bracket no disponible: {e}")

# ---------------------------------------------------------------- tab 3
with tab_estado:
    matches, sync = _load_matches()
    if matches.empty:
        st.warning("Base de datos vacía - corre `python -m scripts.sync_wc2026`.")
    else:
        played = matches[matches["status"] != "NS"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos en base", len(matches))
        c2.metric("Jugados", len(played))
        c3.metric(
            "Último sync",
            f"{sync['source']} · {sync['synced_at'][:10]}" if sync else "-",
        )
        round_names = {
            "group": "Fase de grupos", "round_of_32": "Dieciseisavos",
            "round_of_16": "Octavos", "quarterfinal": "Cuartos",
            "semifinal": "Semifinales", "third_place": "Tercer puesto", "final": "Final",
        }
        sel = st.selectbox("Ronda", list(round_names), format_func=round_names.get,
                           index=2)
        sub = matches[matches["round"] == sel].copy()
        sub["marcador"] = sub.apply(
            lambda r: f"{int(r.home_goals)} - {int(r.away_goals)}"
            if pd.notna(r.home_goals) else "por jugar", axis=1,
        )
        sub["penales"] = sub["penalty_winner"].fillna("")
        st.dataframe(
            sub[["date", "home_team", "marcador", "away_team", "penales", "status", "source"]],
            use_container_width=True, height=480,
        )

# ---------------------------------------------------------------- tab 4
with tab_metodo:
    st.markdown("""
### Metodología

**Datos** · +25,000 partidos internacionales desde 1872
([martj42/international_results](https://github.com/martj42/international_results)) +
ranking FIFA oficial. Los partidos del Mundial 2026 en curso se sincronizan a SQLite
(`data/database.db`) con cascada de fuentes: API-Football → martj42 → CSV manual
(el free tier de API-Football no cubre la temporada 2026; el sistema lo detecta y hace fallback).

**Features** (walk-forward, sin fuga de datos): Elo (k=32 competitivo / 16 amistosos),
diferencia de ranking FIFA, forma reciente (últimos 5), head-to-head histórico,
tasa de victoria en penales, e interacciones.

**Modelo W/D/L** · `GradientBoostingClassifier` (scikit-learn) con **split temporal**
(train ≤2021, calibración 2022-24, test 2025+) y **calibración de probabilidades**
(`CalibratedClassifierCV`; sigmoid elegido por Brier score frente a isotonic).

**Marcador exacto** · dos `PoissonRegressor` (goles esperados λ de cada equipo) +
ajuste **Dixon-Coles** (ρ ajustado por máxima verosimilitud) para la dependencia en
marcadores bajos. El marcador mostrado es el argmax de la matriz **condicionado** al
resultado del clasificador, para que ambos outputs nunca se contradigan.

**Monte Carlo** · el resultado se muestrea de las probabilidades calibradas
(multinomial) y el marcador de la matriz de Poisson condicionada; las tablas de grupos
usan goles simulados reales. Dos modos: torneo completo y **desde el estado actual**
(resultados reales fijos + bracket oficial restante).
""")

    cal_fig = FIGURES_DIR / "calibration_curve.png"
    if cal_fig.exists():
        st.subheader("Curvas de calibración (test temporal 2025+)")
        st.image(str(cal_fig))
    cal_report = OUTPUTS_DIR / "calibration_report.csv"
    if cal_report.exists():
        st.dataframe(pd.read_csv(cal_report), use_container_width=True)

    bt_path = OUTPUTS_DIR / "backtest_results.csv"
    if bt_path.exists():
        st.subheader("Backtest walk-forward (ventana expansiva por año)")
        st.dataframe(pd.read_csv(bt_path), use_container_width=True)
        bt_fig = FIGURES_DIR / "backtest_metrics.png"
        if bt_fig.exists():
            st.image(str(bt_fig))

    st.markdown("""
### Fuentes de datos

| Fuente | Uso |
|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | Histórico de partidos internacionales (~25,400 desde 2000) |
| Ranking FIFA oficial | Fuerza relativa de las selecciones |
| API-Football (api-sports.io) + CSV manual | Resultados del Mundial 2026 en curso |
| Dixon & Coles (1997) | Concepto del ajuste al modelo Poisson para marcadores |
""")
