"""ReverseMilk – app Streamlit (usa agendamento, indicadores, gerar_dados e analises)."""
import os
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agendamento import BUFFER_MIN, agendamento_reverso, reprogramar
from analises import (atraso_por_posicao, avaliar, causa_principal, conclusoes,
                      planejado_vs_real, ranking, sensibilidade)
from gerar_dados import ROTEIROS, gerar
from indicadores import classificar_etapas, distribuicao_mc, resumo_ops

st.set_page_config(page_title="ReverseMilk – Planejamento Reverso e Simulação de Cenários Produtivos",
                   page_icon="🥛", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp {background-color:#12161F; color:#E0E6ED;}
[data-testid="stSidebar"] {background-color:#0A0D14; border-right:1px solid #1E2638;}
.kpi-card {background:#1A2130; border-radius:10px; padding:14px 18px; border-left:4px solid #1E88E5;
           box-shadow:0 4px 12px rgba(0,0,0,.3); margin-bottom:10px;}
.kpi-title {font-size:.8rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:.5px;}
.kpi-value {font-size:1.7rem; font-weight:700; color:#FFF; margin-top:4px;}
.kpi-sub {font-size:.8rem; color:#38BDF8; margin-top:2px;}
.kpi-crit {border-left-color:#EF4444;} .kpi-ok {border-left-color:#10B981;} .kpi-warn {border-left-color:#F59E0B;}
.stTabs [data-baseweb="tab-list"] {gap:8px; background:#0A0D14; padding:8px; border-radius:8px;}
.stTabs [data-baseweb="tab"] {height:42px; background:#1A2130; border-radius:6px; color:#94A3B8; font-weight:600;}
.stTabs [aria-selected="true"] {background:#1E88E5 !important; color:#FFF !important;}
</style>
""", unsafe_allow_html=True)

CORES_STATUS = {"Atrasada": "#EF4444", "Em risco": "#F59E0B", "No prazo": "#10B981",
                "Concluída no prazo": "#64748B", "Concluída com atraso": "#A855F7"}
COLS = ["op", "produto", "ordem_etapa", "etapa", "recurso", "dur_padrao_min", "cv",
        "plan_inicio", "plan_fim", "prazo", "real_inicio", "real_fim"]
DATAS = ["plan_inicio", "plan_fim", "prazo", "real_inicio", "real_fim"]


# ----------------------------- helpers -----------------------------
def kpi(titulo, valor, sub="", cls=""):
    st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-title">{titulo}</div>'
                f'<div class="kpi-value">{valor}</div><div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True)


def tema(fig, altura=None):
    fig.update_layout(paper_bgcolor="#12161F", plot_bgcolor="#12161F", font_color="#E0E6ED",
                      margin=dict(l=10, r=10, t=50, b=10), height=altura)
    return fig


def mostrar(fig):
    """Compatível com versões novas e antigas do Streamlit."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def tabela(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def fmt(ts):
    return pd.Timestamp(ts).strftime("%d/%m %H:%M")


@st.cache_data(show_spinner=False)
def dados_sinteticos(n, seed):
    return gerar(n_ops=n, seed=seed, base=datetime(2026, 10, 5, 6, 0))


@st.cache_data(show_spinner=False)
def calcular_resumo(d, agora, limite, sims):
    return resumo_ops(d, agora, limite, n_sim=sims)


@st.cache_data(show_spinner=False)
def calcular_sens(d, agora, limite, sims, extra):
    return sensibilidade(d, agora, limite, sims, extra_min=extra)


@st.cache_data(show_spinner=False)
def calcular_avaliar(d, agora, limite, sims, mult_cv, extra_min, recurso, prazo_min):
    return avaliar(d, agora, limite, sims, mult_cv=mult_cv, extra_min=extra_min,
                   recurso=recurso, prazo_min=prazo_min)


# ----------------------------- sidebar -----------------------------
topo = st.sidebar.container()
with st.sidebar:
    st.markdown("---")
    modo = st.radio("Modo de visualização", ["Análise completa", "Operador (simples)"])
    arq = st.file_uploader("Usar CSV próprio (opcional)", type=["csv"])
    n_ops = st.number_input("Nº de ordens (dados sintéticos)", 10, 100, 40, step=5)
    seed_dados = st.number_input("Semente dos dados", value=42, step=1)
    tol = st.slider("Tolerância de atraso por etapa (min)", 0, 120, 30)
    limite = st.slider("Folga mínima para não alertar (min)", 0, 180, 60)
    with st.expander("⚙️ Avançado"):
        sims = st.selectbox("Iterações Monte Carlo", [500, 1000, 2000, 5000], index=1)
        seed_mc = st.number_input("Semente Monte Carlo", value=7, step=1)

df = None
fonte = "dados sintéticos"
if arq is not None:
    try:
        df = pd.read_csv(arq, parse_dates=DATAS)
        faltam = [c for c in COLS if c not in df.columns]
        if faltam:
            st.sidebar.error(f"CSV sem as colunas: {', '.join(faltam)}. Usando dados sintéticos.")
            df = None
        else:
            fonte = "CSV próprio"
    except Exception as e:
        st.sidebar.error(f"Não consegui ler o CSV ({e}). Usando dados sintéticos.")
if df is None:
    df = dados_sinteticos(int(n_ops), int(seed_dados))

ini = df.plan_inicio.min().floor("h").to_pydatetime()
fim = df.prazo.max().ceil("h").to_pydatetime()
padrao = min(ini + timedelta(hours=30), fim)
with topo:
    if os.path.exists("logo.png"):
        st.image("logo.png")
    else:
        st.title("🥛 ReverseMilk")
    agora = st.slider("Horário atual (simulado)", min_value=ini, max_value=fim, value=padrao,
                      step=timedelta(minutes=30), format="DD/MM HH:mm")

# ----------------------------- cálculo central -----------------------------
d = classificar_etapas(df, agora, tol)
r = calcular_resumo(d, agora, limite, int(sims))
ativas = r[~r.status.str.startswith("Concluída")].copy()

st.title("ReverseMilk – Planejamento Reverso e Simulação")
st.caption(f"Status simulado: {pd.Timestamp(agora):%d/%m/%Y às %H:%M} · {fonte} (simulação estocástica)")


def visao_geral(simples=False):
    n_atr = int((ativas.status == "Atrasada").sum())
    n_ris = int((ativas.status == "Em risco").sum())
    prob = ativas.prob_atraso.mean() if len(ativas) else 0.0
    c = st.columns(4)
    with c[0]: kpi("Ordens ativas", len(ativas), f"{len(r) - len(ativas)} já concluídas")
    with c[1]: kpi("Atrasadas", n_atr, "Projeção passa do prazo", "kpi-crit")
    with c[2]: kpi("Em risco", n_ris, f"Folga < {limite} min", "kpi-warn")
    with c[3]: kpi("Prob. média de atraso", f"{prob:.0%}", "Monte Carlo, ordens ativas", "kpi-ok")

    if n_atr:
        st.error(f"🚨 {n_atr} ordem(ns) com atraso projetado. Veja a tabela abaixo.")
    elif n_ris:
        st.warning(f"⚠️ {n_ris} ordem(ns) com folga abaixo de {limite} min.")
    else:
        st.success("✅ Todas as ordens ativas estão dentro da folga mínima.")

    if not simples:
        for f in conclusoes(d, r):
            st.markdown(f"- {f}")
        if len(ativas):
            top = ativas.sort_values("prob_atraso", ascending=False).head(10).copy()
            top["prob_pct"] = top.prob_atraso * 100
            fig = px.bar(top, x="prob_pct", y="op", orientation="h", color="status",
                         color_discrete_map=CORES_STATUS, text=top.prob_pct.round(0).astype(int).astype(str) + "%",
                         title="Top 10 ordens por probabilidade de atraso (%)",
                         labels={"prob_pct": "Prob. de atraso (%)", "op": ""})
            fig.update_yaxes(autorange="reversed")
            mostrar(tema(fig))

    if len(ativas):
        t = ativas.sort_values("prob_atraso", ascending=False)[
            ["op", "produto", "etapa_atual", "status", "prob_atraso", "margem_min", "projecao_conclusao", "prazo"]].copy()
        t["prob_atraso"] = (t.prob_atraso * 100).round(0).astype(int).astype(str) + "%"
        t["margem_min"] = t.margem_min.round(0).astype(int)
        t["projecao_conclusao"] = t.projecao_conclusao.map(fmt)
        t["prazo"] = t.prazo.map(fmt)
        tabela(t.rename(columns={"etapa_atual": "etapa atual", "prob_atraso": "prob. atraso",
                                 "margem_min": "margem (min)", "projecao_conclusao": "projeção"}))
    else:
        st.info("Nenhuma ordem ativa neste momento. Mova o horário atual na barra lateral.")


if modo.startswith("Operador"):
    visao_geral(simples=True)
    st.stop()

(tab_vis, tab_tl, tab_mc, tab_rev, tab_wi, tab_garg, tab_dados) = st.tabs(
    ["🚦 Visão geral", "📊 Linha do tempo", "🎲 Monte Carlo", "⏪ Agendamento reverso",
     "⚡ Simular atraso (E se...?)", "🔍 Gargalos e atrasos", "📋 Dados e premissas"])

# ----------------------------- 1. visão geral -----------------------------
with tab_vis:
    visao_geral()

# ----------------------------- 2. linha do tempo + Monte Carlo -----------------------------
with tab_tl:
    ordem_ops = ativas.sort_values("prob_atraso", ascending=False).op.tolist()
    ordem_ops += [o for o in r.op if o not in ordem_ops]
    op_sel = st.selectbox("Ordem de produção", ordem_ops)
    g = d[d.op == op_sel].sort_values("ordem_etapa")
    prazo = g.prazo.iloc[0]
    n_ok = int((g.situacao == "Concluída").sum())
    st_op = r.loc[r.op == op_sel, "status"].iloc[0]
    st.progress(n_ok / len(g), text=f"Progresso: {n_ok} de {len(g)} etapas concluídas ({st_op})")

    plan = pd.DataFrame({"etapa": g.etapa, "inicio": g.plan_inicio, "fim": g.plan_fim, "Tipo": "Planejado"})
    ex = g[g.situacao != "Pendente"]
    real = pd.DataFrame({"etapa": ex.etapa, "inicio": ex.real_inicio,
                         "fim": ex.real_fim.where(ex.real_fim <= agora, pd.Timestamp(agora)), "Tipo": "Realizado"})
    gantt = pd.concat([plan, real], ignore_index=True)
    fig = px.timeline(gantt, x_start="inicio", x_end="fim", y="etapa", color="Tipo",
                      color_discrete_map={"Planejado": "#1E88E5", "Realizado": "#F59E0B"},
                      category_orders={"etapa": list(g.etapa)}, title=f"Planejado x real – {op_sel}")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group")
    for x, nome, cor in [(agora, "agora", "#38BDF8"), (prazo, "prazo", "#EF4444")]:
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=cor, dash="dash"))
        fig.add_annotation(x=x, y=1.02, yref="paper", text=nome, showarrow=False, font=dict(color=cor))
    mostrar(tema(fig, 420))

# ----------------------------- Monte Carlo -----------------------------
with tab_mc:
    st.header("🎲 Monte Carlo: quando esta ordem deve terminar?")
    if not ordem_ops:
        st.info("Não há ordens para simular.")
    else:
        op_mc = st.selectbox("Ordem de produção", ordem_ops, key="mc_op")
        prazo_mc = d[d.op == op_mc].prazo.iloc[0]
        t0, minutos = distribuicao_mc(d, agora, op_mc, buffer=BUFFER_MIN, n_sim=int(sims), seed=int(seed_mc))
        if t0 is None:
            st.info("Esta ordem já foi concluída. Escolha uma ordem ativa (as de maior risco aparecem primeiro) "
                    "ou volte o horário atual na barra lateral.")
        else:
            t0 = pd.Timestamp(t0)
            fins = t0 + pd.to_timedelta(minutos, unit="m")
            prob_mc = float((fins > prazo_mc).mean())
            p50 = t0 + timedelta(minutes=float(np.percentile(minutos, 50)))
            p90 = t0 + timedelta(minutes=float(np.percentile(minutos, 90)))
            m1, m2, m3 = st.columns(3)
            m1.metric("Prob. de atraso", f"{prob_mc:.0%}")
            m2.metric("Conclusão mais provável (P50)", fmt(p50))
            m3.metric("Cenário pessimista (P90)", fmt(p90))
            hist = go.Figure(go.Histogram(x=fins.to_pydatetime(), nbinsx=40, marker_color="#7EC8FF"))
            hist.add_shape(type="line", x0=prazo_mc, x1=prazo_mc, y0=0, y1=1, yref="paper",
                           line=dict(color="#EF4444", dash="dash", width=2))
            hist.add_annotation(x=prazo_mc, y=1.04, yref="paper", text="Prazo", showarrow=False,
                                font=dict(color="#E0E6ED", size=11))
            hist.update_layout(xaxis_title="Conclusão simulada", yaxis_title="Simulações", bargap=0.02)
            tema(hist, 420)
            hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            hist.update_yaxes(gridcolor="#2A2F3A")
            hist.update_xaxes(gridcolor="#2A2F3A")
            mostrar(hist)
            st.caption(f"Simulação a partir de {fmt(t0)} ({int(sims)} cenários). "
                       "Barras à direita da linha do prazo são cenários em que a ordem atrasa.")

# ----------------------------- 3. agendamento reverso -----------------------------
with tab_rev:
    st.caption("Parte do prazo de entrega e calcula, de trás para frente, o horário-limite de cada etapa.")
    a, b, c3, c4 = st.columns(4)
    prod = a.selectbox("Produto", list(ROTEIROS))
    dia = b.date_input("Data de entrega", datetime(2026, 10, 6))
    hora = c3.time_input("Hora de entrega", time(18, 0))
    buf = c4.number_input("Folga entre etapas (min)", 0, 60, BUFFER_MIN)
    prazo_rev = datetime.combine(dia, hora)
    dfp = pd.DataFrame(agendamento_reverso(prazo_rev, ROTEIROS[prod], buf))
    inicio_max = dfp.plan_inicio.min()
    total = (prazo_rev - inicio_max).total_seconds() / 60
    k1, k2 = st.columns(2)
    with k1: kpi("Início máximo permitido", inicio_max.strftime("%d/%m/%Y %H:%M"), "Momento limite para iniciar a 1ª etapa", "kpi-ok")
    with k2: kpi("Duração total", f"{total:.0f} min ({total / 60:.1f} h)", "Com folgas entre etapas")
    fig = px.timeline(dfp, x_start="plan_inicio", x_end="plan_fim", y="etapa", color="recurso",
                      category_orders={"etapa": list(dfp.etapa)}, title="Cronograma retroativo")
    fig.update_yaxes(autorange="reversed")
    mostrar(tema(fig, 380))
    tv = dfp[["etapa", "recurso", "dur_padrao_min", "plan_inicio", "plan_fim"]].copy()
    tv["plan_inicio"] = tv.plan_inicio.map(fmt)
    tv["plan_fim"] = tv.plan_fim.map(fmt)
    tabela(tv.rename(columns={"dur_padrao_min": "duração (min)", "plan_inicio": "início limite", "plan_fim": "fim limite"}))

# ----------------------------- 4. what-if -----------------------------
with tab_wi:
    st.subheader("A) Reprogramar uma ordem quando uma etapa atrasa")
    x1, x2, x3 = st.columns(3)
    op_w = x1.selectbox("Ordem", ordem_ops, key="w_op")
    gw = d[d.op == op_w].sort_values("ordem_etapa")
    etapa_w = x2.selectbox("Etapa que atrasa", list(gw.etapa), index=min(3, len(gw) - 1))
    atraso_w = x3.slider("Atraso adicional (min)", 0, 300, 120, step=15)
    ordem_w = int(gw.loc[gw.etapa == etapa_w, "ordem_etapa"].iloc[0])
    novo = reprogramar(gw[["ordem_etapa", "etapa", "plan_inicio", "plan_fim"]], ordem_w, atraso_w)
    prazo_w = gw.prazo.iloc[0]
    fim_novo = novo.novo_fim.max()
    folga_w = (prazo_w - fim_novo).total_seconds() / 60
    k1, k2 = st.columns(2)
    with k1: kpi("Nova conclusão projetada", fim_novo.strftime("%d/%m %H:%M"), f"Prazo: {fmt(prazo_w)}",
                 "kpi-crit" if folga_w < 0 else "kpi-ok")
    with k2: kpi("Margem sobre o prazo", f"{folga_w:+.0f} min", "negativa = estouro do prazo",
                 "kpi-crit" if folga_w < 0 else "kpi-ok")
    tw = novo[["etapa", "plan_inicio", "novo_inicio", "plan_fim", "novo_fim", "deslocamento_min"]].copy()
    for col in ["plan_inicio", "novo_inicio", "plan_fim", "novo_fim"]:
        tw[col] = tw[col].map(fmt)
    tabela(tw)

    st.subheader("B) Cenários sobre todas as ordens ativas")
    y1, y2, y3, y4 = st.columns(4)
    mult = y1.slider("Variabilidade (× CV)", 0.5, 2.0, 1.0, 0.1)
    extra = y2.slider("Tempo extra por etapa (min)", 0, 120, 30, 5)
    rec = y3.selectbox("Recurso afetado", ["(nenhum)"] + sorted(d.recurso.unique()))
    dprazo = y4.slider("Ajuste de prazo (min)", -240, 240, 0, 15)
    base_av = calcular_avaliar(d, agora, limite, int(sims), 1.0, 0.0, None, 0.0)
    cen_av = calcular_avaliar(d, agora, limite, int(sims), mult, float(extra),
                              None if rec == "(nenhum)" else rec, float(dprazo))
    m = st.columns(4)
    m[0].metric("Prob. média de atraso", f"{cen_av['prob_media']:.1%}", f"{(cen_av['prob_media'] - base_av['prob_media']) * 100:+.1f} pp", delta_color="inverse")
    m[1].metric("Atrasadas", cen_av["atrasadas"], cen_av["atrasadas"] - base_av["atrasadas"], delta_color="inverse")
    m[2].metric("Em risco", cen_av["em_risco"], cen_av["em_risco"] - base_av["em_risco"], delta_color="inverse")
    m[3].metric("Margem média (min)", f"{cen_av['margem_media_min']:.0f}", f"{cen_av['margem_media_min'] - base_av['margem_media_min']:+.0f}")
    comp = pd.DataFrame({"Cenário": ["Base", "Simulado"], "Prob. média de atraso (%)": [base_av["prob_media"] * 100, cen_av["prob_media"] * 100]})
    fig = px.bar(comp, x="Cenário", y="Prob. média de atraso (%)", color="Cenário",
                 color_discrete_sequence=["#1E88E5", "#EF4444"], text_auto=".1f")
    mostrar(tema(fig, 320))

# ----------------------------- 5. gargalos -----------------------------
with tab_garg:
    rk_rec, rk_eta = ranking(d, "recurso"), ranking(d, "etapa")
    if rk_rec.empty:
        st.info("Ainda não há etapas concluídas. Avance o horário atual na barra lateral.")
    else:
        t, e = rk_rec.iloc[0], rk_eta.iloc[0]
        g1, g2, g3 = st.columns(3)
        with g1: kpi("Recurso mais crítico", t.recurso, f"{t.pct_da_perda:.0%} dos minutos perdidos", "kpi-crit")
        with g2: kpi("Etapa que mais perde tempo", e.etapa, f"Estourou o padrão em {e.pct_estouro:.0%} das execuções", "kpi-warn")
        with g3: kpi("Minutos perdidos (total)", f"{rk_rec.perda_total_min.sum():,.0f}".replace(",", "."), "Estouro de ciclo + espera acima do buffer")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(rk_rec.head(10), x="perda_total_min", y="recurso", orientation="h",
                         title="Minutos perdidos por recurso", labels={"perda_total_min": "min", "recurso": ""})
            fig.update_yaxes(autorange="reversed")
            mostrar(tema(fig))
        with c2:
            cp = causa_principal(d, r)
            if cp.empty:
                st.info("Nenhuma ordem atrasada ainda para atribuir causa.")
            else:
                fig = px.bar(cp, x="ordens", y="recurso", orientation="h", title="Causa principal das ordens atrasadas")
                fig.update_yaxes(autorange="reversed")
                mostrar(tema(fig))

        st.subheader("Sensibilidade: e se cada recurso demorar mais?")
        extra_s = st.slider("Tempo extra por etapa (min)", 10, 120, 30, 10, key="extra_sens")
        sens, base_s = calcular_sens(d, agora, limite, int(sims), extra_s)
        fig = px.bar(sens, x="aumento_pp", y="recurso", orientation="h", color="aumento_pp",
                     color_continuous_scale="Reds", title=f"Aumento da prob. média de atraso (pp) com +{extra_s} min",
                     labels={"aumento_pp": "pontos percentuais", "recurso": ""})
        fig.update_yaxes(autorange="reversed")
        mostrar(tema(fig))

        c3, c4 = st.columns(2)
        with c3:
            pv = planejado_vs_real(d).melt("etapa", var_name="tipo", value_name="min")
            mostrar(tema(px.bar(pv, x="etapa", y="min", color="tipo", barmode="group", title="Duração planejada x real (etapas concluídas)")))
        with c4:
            ap = atraso_por_posicao(d)
            mostrar(tema(px.line(ap, x="ordem_etapa", y=["perda_media", "atraso_medio"], markers=True, title="Perda e atraso acumulado por posição na rota")))

# ----------------------------- 6. dados e premissas -----------------------------
with tab_dados:
    st.markdown(f"""
**Premissas do modelo**
- **Agendamento reverso:** parte do prazo e recua etapa a etapa, com folga de {BUFFER_MIN} min entre etapas.
- **Dados sintéticos:** duração de cada etapa = padrão × ruído log-normal (média 1, CV por etapa); 30% das ordens começam atrasadas; 3% das etapas têm parada de 30 a 120 min.
- **Monte Carlo:** para cada ordem, simula só o que ainda não aconteceu (etapa em execução + pendentes), com piso de 10% da duração padrão e folga entre etapas. Chance de atraso = fração dos cenários que passa do prazo; P50/P90 = percentis do tempo restante.
- **Status:** *Atrasada* (projeção após o prazo), *Em risco* (margem menor que {limite} min), *No prazo*.
- **Minutos perdidos:** estouro do tempo padrão + espera acima do buffer, só em etapas concluídas.
""")
    tabela(r)
    st.download_button("📥 Baixar resumo das ordens (CSV)", r.to_csv(index=False).encode("utf-8"),
                       file_name="reversemilk_resumo_ordens.csv", mime="text/csv")
    st.download_button("📥 Baixar etapas (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="reversemilk_etapas.csv", mime="text/csv")
