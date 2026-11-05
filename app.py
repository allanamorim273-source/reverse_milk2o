"""ReverseMilk – Planejamento Reverso e Simulação de Cenários Produtivos."""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analises as an
from agendamento import BUFFER_MIN, agendamento_reverso, reprogramar
from gerar_dados import ROTEIROS, gerar
from indicadores import classificar_etapas, distribuicao_mc, resumo_ops

# Cores da Paleta Dark Executiva & Logo
AZUL, LARANJA = "#1E88E5", "#FF9800"
AZUL_NEON = "#00E5FF"
COR = {
    "No prazo": "#2E7D32",
    "Em risco": "#F57C00",
    "Atrasada": "#D32F2F",
    "Concluída no prazo": "#66BB6A",
    "Concluída com atraso": "#EF5350",
}
BASE = datetime(2026, 10, 5, 6, 0)
COL_DATAS = ["plan_inicio", "plan_fim", "prazo", "real_inicio", "real_fim"]
COL_OBRIG = [
    "op",
    "produto",
    "ordem_etapa",
    "etapa",
    "recurso",
    "dur_padrao_min",
    "cv",
] + COL_DATAS

st.set_page_config(page_title="ReverseMilk", page_icon="🥛", layout="wide")

# ---------------------------------------------------------------- ESTILIZAÇÃO CSS COMPLETA
st.markdown(
    """
    <html lang="pt">
    <head>
        <meta name="google" content="notranslate">
    </head>
    <style>
        /* ==================== BARRA LATERAL (SIDEBAR) CUSTOMIZADA ==================== */
        [data-testid="stSidebar"] {
            background-color: #0A0D14 !important;
            border-right: 1px solid #1E293B !important;
        }

        /* Rótulos e Textos na Sidebar */
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #E2E8F0 !important;
        }

        /* Sliders na Sidebar - Trilha e Marcadores em Azul Tecnológico */
        [data-testid="stSidebar"] div[data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"],
        [data-testid="stSidebar"] div[data-baseweb="slider"] div[style*="background-color: red"] {
            background-color: #1E88E5 !important;
        }
        
        [data-testid="stSidebar"] div[role="slider"] {
            background-color: #00E5FF !important;
            border: 2px solid #1E88E5 !important;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.6) !important;
        }

        /* Valores numéricos vermelhos dos Sliders na Sidebar */
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] + div [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] div[data-testid="stSliderTickBar"] + div {
            color: #00E5FF !important;
            font-weight: 700 !important;
        }

        /* Radio Buttons na Sidebar (Modo de visualização) */
        [data-testid="stSidebar"] div[role="radiogroup"] label div[first-child="true"] {
            border-color: #1E88E5 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {
            background-color: #1E88E5 !important;
            border-color: #00E5FF !important;
        }

        /* Caixa de Upload na Sidebar */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background-color: #12161F !important;
            border: 1px dashed #1E88E5 !important;
            border-radius: 10px !important;
            padding: 8px !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background-color: #1E88E5 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
            background-color: #00E5FF !important;
            color: #000000 !important;
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.5) !important;
        }

        /* Inputs Numéricos e Selects na Sidebar */
        [data-testid="stSidebar"] div[data-baseweb="input"] {
            background-color: #12161F !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
        }

        /* ==================== PAINEL PRINCIPAL & CARDS ==================== */
        .card-box {
            background-color: #12161F;
            border-radius: 12px;
            padding: 16px;
            border-left: 5px solid #1E88E5;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }
        .card-box-danger { border-left-color: #D32F2F; }
        .card-box-warning { border-left-color: #F57C00; }
        .card-box-success { border-left-color: #2E7D32; }
        .card-box-info { border-left-color: #0288D1; }

        .card-title {
            color: #8C9EFF;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 4px;
        }
        .card-value {
            color: #FFFFFF;
            font-size: 24px;
            font-weight: 800;
            line-height: 1.1;
        }
        .card-sub {
            color: #9E9E9E;
            font-size: 11px;
            margin-top: 4px;
        }

        .section-header {
            background: linear-gradient(90deg, #1E293B 0%, rgba(30,41,59,0) 100%);
            padding: 8px 14px;
            border-radius: 8px;
            border-left: 4px solid #1E88E5;
            margin: 16px 0 12px 0;
            font-size: 15px;
            font-weight: 700;
            color: #FFFFFF;
        }

        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 10px 16px !important;
        }

        h1 { font-size: 2.2rem !important; font-weight: 800 !important; color: #FFFFFF !important; }
        h2, h3 { color: #E0E0E0 !important; }
        .streamlit-expanderHeader { font-weight: 600; color: #B0BEC5; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_card(title, value, sub="", style="card-box"):
    """Gera um card HTML estilizado para métricas executivas."""
    return f"""<div class="{style}">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>"""


def estilar_grafico(fig, titulo=None):
    """Aplica o tema Dark Executivo padronizado a qualquer gráfico Plotly."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12, color="#E0E0E0"),
        title=dict(text=titulo, font=dict(size=15, color="#FFFFFF")) if titulo else None,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="#1E293B",
            zerolinecolor="#1E293B",
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1E293B",
            zerolinecolor="#1E293B",
            tickfont=dict(color="#94A3B8"),
        ),
        legend=dict(
            bgcolor="rgba(18, 22, 31, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
            font=dict(color="#E0E0E0"),
        ),
    )
    return fig


def fmt(t):
    return t.strftime("%d/%m %H:%M")


def dur(m):
    h, mm = divmod(abs(round(m)), 60)
    return f"{h}h{mm:02d}" if h else f"{mm} min"


def planejado_vs_real(d):
    """Duração planejada x realizada, média por etapa."""
    c = d[d.situacao == "Concluída"].copy()
    c["realizado"] = c.dur_padrao_min + c.desvio_ciclo_min
    return (
        c.groupby("etapa")
        .agg(
            planejado=("dur_padrao_min", "mean"),
            realizado=("realizado", "mean"),
        )
        .reset_index()
    )


def mostrar_alertas(af, d, simples, risco):
    st.markdown('<div class="section-header">🚨 Alertas Preventivos</div>', unsafe_allow_html=True)
    st.caption(
        f"🔴 Atraso previsto: a projeção passa do prazo · 🟡 Atenção: folga menor que {risco} min."
    )
    al = af[af.status != "No prazo"].copy()
    if al.empty:
        st.success("Tudo no prazo por enquanto. Nenhum alerta registrado.")
        return
    al["tipo"] = (al.status != "Atrasada").astype(int)
    al = al.sort_values(["tipo", "prob_atraso"], ascending=[True, False])

    def linha(o):
        if simples:
            pend = d[(d.op == o.op) & (d.situacao != "Concluída")].sort_values(
                "ordem_etapa"
            )
            onde = pend.iloc[0].recurso if len(pend) else "—"
            sit = (
                f"{dur(o.margem_min)} de atraso"
                if o.margem_min < 0
                else f"só {dur(o.margem_min)} de folga"
            )
            txt = (
                f"**{o.op}** ({o.produto}) · etapa: {o.etapa_atual} · termina {fmt(o.projecao_conclusao)}, "
                f"prazo {fmt(o.prazo)} · {sit} · onde agir: **{onde}**"
            )
        else:
            txt = (
                f"**{o.op}** ({o.produto}) · etapa: {o.etapa_atual} · projeção {fmt(o.projecao_conclusao)} × "
                f"prazo {fmt(o.prazo)} · margem {o.margem_min:.0f} min · chance de atraso {o.prob_atraso:.0%}"
            )
        if o.status == "Atrasada":
            st.error(txt, icon="🔴")
        else:
            st.warning(txt, icon="🟡")

    for _, o in al.head(5).iterrows():
        linha(o)
    if len(al) > 5:
        with st.expander(f"Ver mais {len(al) - 5} alertas de produção"):
            for _, o in al.iloc[5:].iterrows():
                linha(o)


def rodape():
    st.divider()
    st.caption(
        "Projeto acadêmico SENAI · Centro 4.0. A Piracanjuba é apenas a empresa de referência do desafio; "
        "este protótipo usa dados sintéticos e não tem vínculo nem validação da empresa."
    )


@st.cache_data
def gerar_cache(n, seed):
    return gerar(n_ops=n, seed=seed, base=BASE)


@st.cache_data
def calcular(df, agora, tol, risco, sims):
    d = classificar_etapas(df, agora, tol)
    return d, resumo_ops(d, agora, risco, n_sim=sims)


@st.cache_data
def sens(d, agora, risco, sims):
    return an.sensibilidade(d, agora, risco, sims)


# ---------------------------------------------------------------- BARRA LATERAL (SIDEBAR)
with st.sidebar:
    if Path("logo.png").exists():
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("## 🥛 **REVERSE MILK**")

    st.markdown("---")
    modo = st.radio(
        "Modo de visualização",
        ["Análise completa", "Operador (simples)"],
        help="Operador: só alertas e acompanhamento das ordens. Análise completa: todos os gráficos e cenários.",
    )
    OP = modo.startswith("Operador")
    caixa_horario = st.container()

    if OP:
        arq, n_ops, tol, risco, seed, sims = None, 40, 30, 60, 42, 1000
    else:
        arq = st.file_uploader("Usar CSV próprio (opcional)", type="csv")
        n_ops = st.number_input("Nº de ordens (dados sintéticos)", 10, 80, 40, 5)
        tol = st.slider(
            "Tolerância de atraso por etapa (min)",
            5,
            90,
            30,
            5,
            help="Até esse atraso a etapa fica 🟡 em risco; acima disso, 🔴 atrasada.",
        )
        risco = st.slider(
            "Folga mínima para não alertar (min)",
            15,
            180,
            60,
            15,
            help="Se a ordem deve terminar com menos folga que isso, ela vira 🟡 atenção.",
        )
        with st.expander("Configurações avançadas"):
            seed = st.number_input(
                "Semente aleatória",
                value=42,
                step=1,
                help="Muda os dados sintéticos. Mesma semente = mesmos resultados.",
            )
            sims = st.select_slider(
                "Simulações Monte Carlo", [500, 1000, 2000, 5000], 1000
            )

# Carregamento de dados
if arq is not None:
    df = pd.read_csv(arq, parse_dates=COL_DATAS)
    faltam = [c for c in COL_OBRIG if c not in df.columns]
    if faltam:
        st.error(f"O CSV não tem as colunas: {', '.join(faltam)}")
        st.stop()
    origem = "dados do CSV enviado"
else:
    df = gerar_cache(int(n_ops), int(seed))
    origem = "dados sintéticos (simulação estocástica)"

ini = df.plan_inicio.min().floor("h").to_pydatetime()
fim = df.real_fim.max().ceil("h").to_pydatetime()

with caixa_horario:
    agora = st.slider(
        "Horário atual (simulado)",
        min_value=ini,
        max_value=fim,
        value=ini
        + timedelta(hours=int((fim - ini).total_seconds() / 7200)),
        step=timedelta(hours=1),
        format="DD/MM HH:mm",
        help="Move o 'relógio' da simulação. Quanto mais tarde, mais etapas já foram concluídas.",
    )

d, r = calcular(df, agora, tol, risco, sims)
ativas = r[~r.status.str.startswith("Concluída")]

# Cabeçalho Principal Estilizado
st.markdown("# Planejamento Reverso & Cenários")
st.caption(f"Status simulado: {agora:%d/%m/%Y às %H:%M} · {origem}")

if OP:
    t1, t2 = st.tabs(["🚦 Visão geral", "📋 Acompanhar uma ordem"])
    with t1:
        st.markdown(
            f"### 🔴 {int((ativas.status == 'Atrasada').sum())} atraso previsto · "
            f"🟡 {int((ativas.status == 'Em risco').sum())} atenção · "
            f"🟢 {int((ativas.status == 'No prazo').sum())} no prazo"
        )
        mostrar_alertas(ativas, d, True, risco)
    with t2:
        op_o = st.selectbox(
            "Escolha a ordem",
            r.sort_values("prob_atraso", ascending=False).op,
            key="op_operador",
            format_func=lambda o: f"{o} · {r.set_index('op').loc[o, 'produto']}",
        )
        lo = r.set_index("op").loc[op_o]
        ic = {"No prazo": "🟢", "Em risco": "🟡", "Atrasada": "🔴"}.get(
            lo.status, "✅"
        )
        st.markdown(
            f"**{ic} {lo.status}** · termina {fmt(lo.projecao_conclusao)} · prazo {fmt(lo.prazo)}"
        )
        st.progress(
            int(lo.etapas_concluidas) / int(lo.total_etapas),
            text=f"{lo.etapas_concluidas} de {lo.total_etapas} etapas concluídas",
        )
        for _, e in d[d.op == op_o].sort_values("ordem_etapa").iterrows():
            icone = {
                "Concluída": "✅",
                "Em execução": "▶️",
                "Pendente": "⏳",
            }[e.situacao]
            nome = f"**{e.etapa}**" if e.situacao == "Em execução" else e.etapa
            st.markdown(
                f"{icone} {nome} ({e.recurso}) · planejado {fmt(e.plan_inicio)} a {fmt(e.plan_fim)}"
            )
    rodape()
    st.stop()

# Definição das Abas
abas = st.tabs(
    [
        "🚦 Visão geral",
        "📊 Linha do tempo",
        "⏪ Agendamento reverso",
        "⚡ Simular atraso (E se...?)",
        "🔍 Gargalos e Atrasos",
        "📋 Dados e Premissas",
    ]
)

# ---------------------------------------------------------------- ABA 1: VISÃO GERAL
with abas[0]:
    todos = sorted(r.produto.unique())
    prods = st.multiselect("Filtrar por linha de produto", todos, default=todos)
    rf = r[r.produto.isin(prods)]
    af = rf[~rf.status.str.startswith("Concluída")]

    with st.expander("ℹ️ Entenda os critérios de risco e as premissas"):
        st.markdown(f"""
        - **🔴 Atrasada:** A projeção de término da ordem ultrapassa o prazo final estipulado.
        - **🟡 Em risco (Atenção):** A ordem termina dentro do prazo, mas com margem de folga menor que **{risco} min**.
        - **🟢 No prazo:** A ordem possui folga igual ou superior a **{risco} min**.
        - **Simulação Monte Carlo:** Executa **{sims} iterações** probabilísticas sobre as etapas pendentes para estimar a chance real de atraso diante da variabilidade operacional.
        """)

    # Cards de KPIs Executivos
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_card("OPS ATIVAS", len(af), "Em andamento"), unsafe_allow_html=True)
    with col2:
        n_criticas = int((af.status != "No prazo").sum())
        st.markdown(render_card("CRÍTICAS", n_criticas, "Atraso ou Risco", "card-box card-box-danger"), unsafe_allow_html=True)
    with col3:
        pct_sucesso = ((int((af.status == "No prazo").sum()) / len(af) * 100) if len(af) > 0 else 100)
        st.markdown(render_card("SUCESSO", f"{pct_sucesso:.0f}%", "Entregas no prazo", "card-box card-box-success"), unsafe_allow_html=True)
    with col4:
        desvio_m = af["margem_min"].mean() if len(af) > 0 else 0
        txt_desvio = f"{abs(desvio_m):.0f} min" if desvio_m < 0 else f"+{desvio_m:.0f} min"
        st.markdown(render_card("DESVIO MÉDIO", txt_desvio, "Margem média projetada", "card-box card-box-warning"), unsafe_allow_html=True)

    for frase in an.conclusoes(d[d.produto.isin(prods)], rf):
        st.markdown(f"- {frase}")

    mostrar_alertas(af, d, False, risco)

    st.markdown('<div class="section-header">⚠️ Ordens de Maior Risco (Top 5)</div>', unsafe_allow_html=True)
    top5_alertas = af.sort_values(by="prob_atraso", ascending=False).head(5)

    if not top5_alertas.empty:
        disp_top5 = top5_alertas.copy()
        disp_top5["Projeção"] = disp_top5["projecao_conclusao"].apply(fmt)
        disp_top5["Prazo"] = disp_top5["prazo"].apply(fmt)

        st.dataframe(
            disp_top5[
                [
                    "op",
                    "produto",
                    "etapa_atual",
                    "Projeção",
                    "Prazo",
                    "margem_min",
                    "prob_atraso",
                    "status",
                ]
            ],
            column_config={
                "op": "Ordem",
                "produto": "Produto",
                "etapa_atual": "Etapa atual",
                "margem_min": st.column_config.NumberColumn("Margem", format="%d min"),
                "prob_atraso": st.column_config.ProgressColumn(
                    "Prob. de Atraso",
                    help="Simulação Monte Carlo",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "status": "Status",
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("Nenhuma ordem em estado crítico no momento.")

    cont = rf.status.value_counts().reset_index()
    cont.columns = ["status", "ordens"]
    f = px.bar(cont, x="status", y="ordens", color="status", color_discrete_map=COR)
    f.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(estilar_grafico(f, "Distribuição Geral das Ordens de Produção"), use_container_width=True)

    with st.expander("Ver tabela completa de ordens"):
        st.dataframe(rf, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- ABA 2: LINHA DO TEMPO
with abas[1]:
    st.markdown('<div class="section-header">📊 Acompanhamento Sequencial da Ordem</div>', unsafe_allow_html=True)
    op = st.selectbox(
        "Selecione a Ordem de Produção (OP)",
        r.sort_values("prob_atraso", ascending=False).op,
        format_func=lambda o: f"{o} · {r.set_index('op').loc[o, 'produto']}",
    )
    linha = r.set_index("op").loc[op]
    st.progress(
        int(linha.etapas_concluidas) / int(linha.total_etapas),
        text=f"Progresso: {linha.etapas_concluidas} de {linha.total_etapas} etapas concluídas ({linha.status})",
    )
    g = d[d.op == op]
    prazo = g.prazo.iloc[0]
    linhas = []
    for _, e in g.iterrows():
        linhas.append(dict(etapa=e.etapa, ini=e.plan_inicio, fim=e.plan_fim, tipo="Planejado"))
        if e.real_inicio <= agora:
            linhas.append(dict(etapa=e.etapa, ini=e.real_inicio, fim=min(e.real_fim, agora), tipo="Realizado"))

    f = px.timeline(
        pd.DataFrame(linhas),
        x_start="ini",
        x_end="fim",
        y="etapa",
        color="tipo",
        color_discrete_map={"Planejado": "#64B5F6", "Realizado": AZUL},
        category_orders={"etapa": list(g.etapa)},
    )
    f.update_yaxes(autorange="reversed", title=None)
    for x, nome, cor in [(agora, "Agora", "#81D4FA"), (prazo, "Prazo", "#EF5350")]:
        f.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=cor, dash="dash", width=2))
        f.add_annotation(x=x, y=1.04, yref="paper", text=nome, showarrow=False, font=dict(color=cor, size=12))
    st.plotly_chart(estilar_grafico(f, f"Cronograma Gantt – {op} ({linha.produto})"), use_container_width=True)

    t0, mins = distribuicao_mc(d, agora, op, n_sim=sims)
    if t0 is None:
        st.info("Ordem já concluída. Não há etapas pendentes para simular.")
    else:
        fim_s = t0 + pd.to_timedelta(mins, unit="m")
        p50, p90 = fim_s.quantile(0.5), fim_s.quantile(0.9)
        chance_atraso = (fim_s > prazo).mean()

        st.markdown('<div class="section-header">🎲 Análise Probabilística de Risco (Monte Carlo)</div>', unsafe_allow_html=True)
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(render_card("PRAZO FINAL", fmt(prazo), "Horário limite estipulado", "card-box card-box-info"), unsafe_allow_html=True)
        with mc2:
            st.markdown(render_card("MAIS PROVÁVEL (P50)", fmt(p50), "Mediana das simulações"), unsafe_allow_html=True)
        with mc3:
            st.markdown(render_card("SEGURANÇA 90% (P90)", fmt(p90), "9 em cada 10 cenários", "card-box card-box-warning"), unsafe_allow_html=True)
        with mc4:
            st_style = "card-box card-box-danger" if chance_atraso > 0.3 else "card-box card-box-success"
            st.markdown(render_card("CHANCE DE ATRASO", f"{chance_atraso:.0%}", "Probabilidade estocástica", st_style), unsafe_allow_html=True)

        f_hist = px.histogram(x=fim_s, nbins=40, color_discrete_sequence=[AZUL])
        for x, nome, cor in [(prazo, "Prazo", "#EF5350"), (p50, "Mais provável", LARANJA), (p90, "90%", "#B0BEC5")]:
            f_hist.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=cor, dash="dash", width=2))
            f_hist.add_annotation(x=x, y=1.04, yref="paper", text=nome, showarrow=False, font=dict(color=cor))
        f_hist.update_layout(xaxis_title=None, yaxis_title="nº de cenários", bargap=0.05)
        st.plotly_chart(estilar_grafico(f_hist, f"Distribuição Estocástica de Conclusão ({sims} iterações)"), use_container_width=True)

# ---------------------------------------------------------------- ABA 3: AGENDAMENTO REVERSO
with abas[2]:
    st.markdown('<div class="section-header">⏪ Cálculo do Horário-Limite de Início</div>', unsafe_allow_html=True)
    st.caption("Informe o produto e a data/hora final desejada. O algoritmo realiza o encadeamento reverso das etapas descontando tempos padrões e folgas operacionais.")
    
    with st.container():
        a, b, c, e = st.columns(4)
        prod = a.selectbox("Produto Final", list(ROTEIROS))
        dia = b.date_input("Data de Entrega Desejada", datetime(2026, 10, 6).date())
        hora = c.time_input("Hora de Entrega Desejada", datetime(2026, 10, 6, 18, 0).time())
        folga = e.number_input("Folga entre etapas (min)", 0, 120, BUFFER_MIN, 5)

    entrega = datetime.combine(dia, hora)
    pl = pd.DataFrame(agendamento_reverso(entrega, ROTEIROS[prod], folga))
    inicio_limite = pl.plan_inicio.min()
    duracao_total = (entrega - inicio_limite).total_seconds() / 3600

    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(render_card("INÍCIO MÁXIMO PERMITIDO", fmt(inicio_limite), "A produção deve começar até este horário", "card-box card-box-warning"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_card("DURAÇÃO TOTAL ESTIMADA", f"{duracao_total:.1f} horas", "Tempo total do ciclo de produção", "card-box card-box-info"), unsafe_allow_html=True)

    f_rev = px.timeline(pl, x_start="plan_inicio", x_end="plan_fim", y="etapa", color="recurso")
    f_rev.update_yaxes(autorange="reversed", title=None)
    st.plotly_chart(estilar_grafico(f_rev, f"Cronograma Reverso Calculado – {prod}"), use_container_width=True)
    
    with st.expander("Ver detalhamento dos horários calculados por etapa"):
        st.dataframe(
            pl[["ordem_etapa", "etapa", "recurso", "dur_padrao_min", "plan_inicio", "plan_fim"]],
            hide_index=True,
            use_container_width=True,
        )

# ---------------------------------------------------------------- ABA 4: SIMULAR ATRASO (E SE...?)
with abas[3]:
    st.markdown('<div class="section-header">⚡ Impacto Instantâneo de Atraso em Etapa</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    op_w = a.selectbox("Selecione a Ordem", list(ativas.op) or list(r.op))
    plano = d[d.op == op_w].sort_values("ordem_etapa")[["ordem_etapa", "etapa", "recurso", "dur_padrao_min", "plan_inicio", "plan_fim"]]
    nomes = plano.set_index("ordem_etapa").etapa
    ordem = b.selectbox("Etapa afetada", list(nomes.index), format_func=lambda i: nomes[i])
    atraso = c.slider("Simular atraso adicional (min)", 0, 480, 120, 15)
    
    rp = reprogramar(plano, ordem, atraso)
    if atraso:
        novo_fim_dt = rp.novo_fim.max()
        prazo_orig_dt = plano.plan_fim.max()
        st.markdown(render_card("NOVO HORÁRIO DE CONCLUSÃO PROJETADO", fmt(novo_fim_dt), f"Atraso acumulado no lote: +{atraso} min (Prazo original: {fmt(prazo_orig_dt)})", "card-box card-box-danger"), unsafe_allow_html=True)
        afet = rp[rp.ordem_etapa > ordem].etapa.tolist()
        if afet:
            st.warning(f"⚠️ **Etapas subsequentes cascateadas:** {', '.join(afet)}")
    else:
        st.success("Sem atraso simulado: o fluxo produtivo permanece no cronograma original.")

    with st.expander("Ver comparação de horários antes x depois da reprogramação"):
        st.dataframe(
            rp[["etapa", "recurso", "plan_inicio", "novo_inicio", "plan_fim", "novo_fim", "deslocamento_min"]],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown('<div class="section-header">🔀 Análise de Cenários Estocásticos Combinados</div>', unsafe_allow_html=True)
    recursos = sorted(d.recurso.unique())
    cc = st.columns(4)
    mult = cc[0].slider("Oscilação de Tempos (×)", 0.5, 3.0, 1.5, 0.1)
    rec_sel = cc[1].selectbox("Recurso Sob Gargalo", recursos, index=recursos.index("Tanque Hidrólise") if "Tanque Hidrólise" in recursos else 0)
    extra = cc[2].slider("Tempo Extra no Recurso (min)", 0, 240, 60, 15)
    prazo_d = cc[3].slider("Ajuste do Prazo Final (min)", -240, 240, -120, 30)

    cen = {
        "Situação Atual": {},
        "Tempos Mais Variáveis": dict(mult_cv=mult),
        f"+{extra} min em {rec_sel}": dict(extra_min=extra, recurso=rec_sel),
        f"Prazo {prazo_d:+d} min": dict(prazo_min=prazo_d),
        "Cenário Combinado": dict(mult_cv=mult, extra_min=extra, recurso=rec_sel, prazo_min=prazo_d),
    }
    res = pd.DataFrame({n: an.avaliar(d, agora, risco, min(sims, 1000), **p) for n, p in cen.items()}).T.reset_index(names="cenário")

    f = px.bar(res, x="cenário", y="prob_media", color_discrete_sequence=[AZUL])
    f.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    st.plotly_chart(estilar_grafico(f, "Probabilidade Média de Atraso por Cenário Simulado"), use_container_width=True)

    pior = res.iloc[1:].sort_values("prob_media").iloc[-1]
    st.info(f"💡 **Conclusão:** O cenário mais severo é **{pior['cenário']}**, elevando a probabilidade média de atraso para **{pior.prob_media:.0%}** (comparado a {res.iloc[0].prob_media:.0%} na situação atual).")

# ---------------------------------------------------------------- ABA 5: GARGALOS E ATRASOS
with abas[4]:
    st.markdown('<div class="section-header">🔍 Diagnóstico de Gargalos e Causa-Raiz</div>', unsafe_allow_html=True)
    rec, eta = an.ranking(d, "recurso"), an.ranking(d, "etapa")
    
    if rec.empty:
        st.info("Ainda não existem etapas concluídas até o horário simulado atual.")
    else:
        top_rec = rec.iloc[0]
        top_eta = eta.iloc[0]

        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(render_card("RECURSO MAIS CRÍTICO", top_rec.recurso, f"{top_rec.perda_total_min:.0f} min acumulados", "card-box card-box-danger"), unsafe_allow_html=True)
        with g2:
            st.markdown(render_card("ETAPA GARGALO", top_eta.etapa, f"{top_eta.pct_estouro:.0%} de estouro sobre o padrão", "card-box card-box-warning"), unsafe_allow_html=True)
        with g3:
            st.markdown(render_card("CONCENTRAÇÃO DE PERDAS", f"{top_rec.pct_da_perda:.0%}", "Do tempo perdido total da fábrica", "card-box card-box-info"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        a, b = st.columns(2)
        f_rec = px.bar(rec.head(8).iloc[::-1], x="perda_total_min", y="recurso", orientation="h", color_discrete_sequence=[LARANJA])
        f_rec.update_layout(xaxis_title="Minutos Perdidos Acumulados", yaxis_title=None)
        a.plotly_chart(estilar_grafico(f_rec, "Perda de Tempo por Recurso"), use_container_width=True)

        f_eta = px.bar(eta.head(8).iloc[::-1], x="perda_total_min", y="etapa", orientation="h", color_discrete_sequence=[AZUL])
        f_eta.update_layout(xaxis_title="Minutos Perdidos Acumulados", yaxis_title=None)
        b.plotly_chart(estilar_grafico(f_eta, "Perda de Tempo por Etapa"), use_container_width=True)

        c1, c2 = st.columns(2)
        ap = an.atraso_por_posicao(d)
        f_pos = go.Figure()
        f_pos.add_bar(x=ap.ordem_etapa, y=ap.perda_media, name="Perda na Etapa (min)", marker_color=LARANJA)
        f_pos.add_scatter(x=ap.ordem_etapa, y=ap.atraso_medio, name="Atraso Acumulado (min)", mode="lines+markers", line=dict(color=AZUL, width=3))
        f_pos.update_layout(xaxis_title="Ordem da Etapa no Fluxo", legend=dict(orientation="h", y=-0.3))
        c1.plotly_chart(estilar_grafico(f_pos, "Formação do Atraso ao Longo da Rota"), use_container_width=True)

        cp = an.causa_principal(d, r)
        if len(cp):
            f_cp = px.bar(cp.iloc[::-1], x="ordens", y="recurso", orientation="h", color_discrete_sequence=[AZUL])
            f_cp.update_layout(xaxis_title="Nº de Ordens Afetadas", yaxis_title=None)
            c2.plotly_chart(estilar_grafico(f_cp, "Recurso Causa-Raiz das Ordens Atrasadas"), use_container_width=True)
        else:
            c2.info("Nenhuma ordem atrasada registrada até o momento.")

    if not rec.empty:
        pr = planejado_vs_real(d).melt(id_vars="etapa", var_name="tipo", value_name="minutos")
        f_pr = px.bar(pr, x="etapa", y="minutos", color="tipo", barmode="group", color_discrete_map={"planejado": "#90CAF9", "realizado": AZUL})
        f_pr.update_layout(xaxis_title=None)
        st.plotly_chart(estilar_grafico(f_pr, "Duração Média Planejada × Realizada (min)"), use_container_width=True)

    st.markdown('<div class="section-header">📈 Análise de Sensibilidade do Prazo por Recurso</div>', unsafe_allow_html=True)
    s, base_p = sens(d, agora, risco, min(sims, 1000))
    f_sens = px.bar(s.head(10).iloc[::-1], x="aumento_pp", y="recurso", orientation="h", color_discrete_sequence=[LARANJA])
    f_sens.update_layout(xaxis_title=f"Aumento na Chance de Atraso (pontos %) · Base Atual: {base_p:.0%}", yaxis_title=None)
    st.plotly_chart(estilar_grafico(f_sens, "Sensibilidade: Impacto de +30 min por Recurso na Chance de Atraso Global"), use_container_width=True)

# ---------------------------------------------------------------- ABA 6: DADOS E PREMISSAS
with abas[5]:
    st.markdown('<div class="section-header">📋 Documentação e Roteiro de Parâmetros</div>', unsafe_allow_html=True)
    st.markdown(f"""
    - **Horário Previsto de Conclusão:** Tempo restante estimado + **{BUFFER_MIN} min** de buffer padrão entre etapas consecutivas.
    - **Distribuição Estocástica (Monte Carlo):** Variação aleatória derivada do Coeficiente de Variação (CV) de cada etapa, simulando **{sims} cenários reais**.
    - **Indicadores P50 / P90:** Correspondem à mediana e ao percentil de 90% da distribuição acumulada de encerramento das ordens.
    """)

    st.markdown('<div class="section-header">⚙️ Coeficientes de Variação (CV) por Produto e Etapa</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame([
            dict(
                produto=p,
                ordem=i + 1,
                etapa=e[0],
                recurso=e[1],
                duracao_min=e[2],
                cv=e[3],
            )
            for p, et in ROTEIROS.items()
            for i, e in enumerate(et)
        ]),
        hide_index=True,
        use_container_width=True,
    )
    
    st.markdown('<div class="section-header">💾 Base de Dados Sintéticos Ativa</div>', unsafe_allow_html=True)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button(
        "Baixar Base de Dados (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="dados_reversemilk.csv",
        mime="text/csv",
    )

rodape()
