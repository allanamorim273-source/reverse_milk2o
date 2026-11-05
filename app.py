"""ReverseMilk – Planeamento Reverso e Simulação de Cenários Produtivos."""
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analises as an
from agendamento import BUFFER_MIN, agendamento_reverso, reprogramar
from gerar_dados import ROTEIROS, gerar
from indicadores import classificar_etapas, distribuicao_mc, resumo_ops

AZUL, LARANJA = "#004B93", "#F47920"
COR = {
    "No prazo": "#2E8B57",
    "Em risco": "#E0A100",
    "Atrasada": "#C62828",
    "Concluída no prazo": "#7FB89A",
    "Concluída com atraso": "#D98C8C",
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

# Previne o erro 'removeChild' impedindo que tradutores automáticos alterem a estrutura do React
st.markdown(
    f"""
    <html lang="pt">
    <head>
        <meta name="google" content="notranslate">
    </head>
    <style>
        h1,h2,h3 {{color:{AZUL};}}
        [data-testid="stMetric"] {{border-left:4px solid {LARANJA};padding-left:12px}}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt(t):
    return t.strftime("%d/%m %H:%M")


def dur(m):
    h, mm = divmod(abs(round(m)), 60)
    return f"{h}h{mm:02d}" if h else f"{mm} min"


def planejado_vs_real(d):
    """Duração planeada (padrão) x realizada, média por etapa, só etapas concluídas."""
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
    st.subheader("🚨 Alertas preventivos")
    st.caption(
        f"🔴 Atraso previsto: a projeção passa do prazo · 🟡 Atenção: folga menor que {risco} min."
    )
    al = af[af.status != "No prazo"].copy()
    if al.empty:
        st.success("Tudo no prazo por enquanto. Nenhum alerta.")
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
        with st.expander(f"Ver mais {len(al) - 5} alertas"):
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


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("### 🥛 ReverseMilk")
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

st.title("ReverseMilk – Planeamento Reverso e Simulação de Cenários Produtivos")
st.caption(f"Horário simulado: {agora:%d/%m/%Y %H:%M} · {origem}")

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

abas = st.tabs(
    [
        "🚦 Visão geral",
        "🔍 Onde estão os atrasos",
        "🔀 E se...?",
        "📋 Acompanhar uma ordem",
        "🧮 Calcular horários",
        "📖 Como funciona e dados",
    ]
)

# ---------------------------------------------------------------- 1. visão geral
with abas[0]:
    todos = sorted(r.produto.unique())
    prods = st.multiselect("Filtrar por produto", todos, default=todos)
    rf = r[r.produto.isin(prods)]
    af = rf[~rf.status.str.startswith("Concluída")]

    # Explicação das premissas e critérios
    with st.expander("ℹ️ Entenda os critérios de risco e as premissas"):
        st.markdown(f"""
        - **🔴 Atrasada:** A projeção de término da ordem ultrapassa o prazo final estipulado[cite: 31].
        - **🟡 Em risco (Atenção):** A ordem termina dentro do prazo, mas com margem de folga menor que **{risco} min**[cite: 31].
        - **🟢 No prazo:** A ordem possui folga igual ou superior a **{risco} min**[cite: 31].
        - **Simulação Monte Carlo:** Executa **{sims} iterações** probabilísticas sobre as etapas pendentes para estimar a chance real de atraso diante da variabilidade operacional[cite: 13, 31].
        """)

    st.markdown(
        "**Como ler:** 🟢 no prazo · 🟡 atenção (pouca folga) · 🔴 atraso previsto."
    )
    c = st.columns(4)
    c[0].metric("Ordens em andamento", len(af))
    c[1].metric("🟢 No prazo", int((af.status == "No prazo").sum()))
    c[2].metric("🟡 Atenção", int((af.status == "Em risco").sum()))
    c[3].metric("🔴 Atraso previsto", int((af.status == "Atrasada").sum()))

    for frase in an.conclusoes(d[d.produto.isin(prods)], rf):
        st.markdown(f"- {frase}")

    mostrar_alertas(af, d, False, risco)

    # Tabela resumida de atenção com barra visual de probabilidade (Top 5)
    st.subheader("⚠️ Ordens de atenção agora (Top 5)")
    st.caption("As 5 ordens ativas com maior probabilidade de atraso.")
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
                "margem_min": st.column_config.NumberColumn(
                    "Margem (min)", format="%d min"
                ),
                "prob_atraso": st.column_config.ProgressColumn(
                    "Prob. de atraso",
                    help="Calculado via simulação de Monte Carlo",
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
    f = px.bar(
        cont,
        x="status",
        y="ordens",
        color="status",
        color_discrete_map=COR,
        title="Ordens por situação",
    )
    f.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(f, use_container_width=True)

    with st.expander("Ver todas as ordens em detalhe"):
        st.dataframe(rf, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 2. onde estão os atrasos
with abas[1]:
    st.caption(
        f"Minutos perdidos = tempo acima do padrão da etapa + espera acima dos {BUFFER_MIN} min previstos "
        "entre etapas. Só conta o que já foi concluído."
    )
    rec, eta = an.ranking(d, "recurso"), an.ranking(d, "etapa")
    if rec.empty:
        st.info(
            "Ainda não há etapas concluídas. Avance o horário atual na barra lateral."
        )
    else:
        top_rec = rec.iloc[0]
        top_eta = eta.iloc[0]

        # KPIs de destaque para aprofundamento de gargalos
        col_g1, col_g2, col_g3 = st.columns(3)
        col_g1.metric(
            "Recurso mais Crítico",
            top_rec.recurso,
            f"{top_rec.perda_total_min:.0f} min perdidos",
        )
        col_g2.metric(
            "Etapa Gargalo",
            top_eta.etapa,
            f"{top_eta.pct_estouro:.0%} de estouro",
        )
        col_g3.metric(
            "Concentração de Perdas",
            f"{top_rec.pct_da_perda:.0%} do total",
            "perdas acumuladas",
        )
        st.divider()

        st.markdown(
            f"**Maior gargalo até agora:** {top_rec.recurso}, com {top_rec.pct_da_perda:.0%} dos minutos "
            f"perdidos ({top_rec.perda_total_min:.0f} min)."
        )
        a, b = st.columns(2)
        f = px.bar(
            rec.head(8).iloc[::-1],
            x="perda_total_min",
            y="recurso",
            orientation="h",
            title="Recursos que mais perdem tempo",
            color_discrete_sequence=[LARANJA],
        )
        f.update_layout(xaxis_title="minutos perdidos", yaxis_title=None)
        a.plotly_chart(f, use_container_width=True)
        f = px.bar(
            eta.head(8).iloc[::-1],
            x="perda_total_min",
            y="etapa",
            orientation="h",
            title="Etapas que mais perdem tempo",
            color_discrete_sequence=[AZUL],
        )
        f.update_layout(xaxis_title="minutos perdidos", yaxis_title=None)
        b.plotly_chart(f, use_container_width=True)

        c1, c2 = st.columns(2)
        ap = an.atraso_por_posicao(d)
        f = go.Figure()
        f.add_bar(
            x=ap.ordem_etapa,
            y=ap.perda_media,
            name="Tempo perdido na etapa (min)",
            marker_color=LARANJA,
        )
        f.add_scatter(
            x=ap.ordem_etapa,
            y=ap.atraso_medio,
            name="Atraso acumulado no fim da etapa (min)",
            mode="lines+markers",
            line=dict(color=AZUL),
        )
        f.update_layout(
            title="Onde o atraso nasce e como se acumula",
            xaxis_title="Posição da etapa na rota",
            legend=dict(orientation="h", y=-0.3),
        )
        c1.plotly_chart(f, use_container_width=True)
        cp = an.causa_principal(d, r)
        if len(cp):
            f = px.bar(
                cp.iloc[::-1],
                x="ordens",
                y="recurso",
                orientation="h",
                title="Recurso que mais atrasou cada ordem atrasada",
                color_discrete_sequence=[AZUL],
            )
            f.update_layout(xaxis_title="nº de ordens", yaxis_title=None)
            c2.plotly_chart(f, use_container_width=True)
        else:
            c2.info(
                "Nenhuma ordem atrasada com perdas registradas até agora."
            )

    if not rec.empty:
        pr = planejado_vs_real(d).melt(
            id_vars="etapa", var_name="tipo", value_name="minutos"
        )
        f = px.bar(
            pr,
            x="etapa",
            y="minutos",
            color="tipo",
            barmode="group",
            color_discrete_map={"planejado": "#9DB8D9", "realizado": AZUL},
            title="Duração planejada × realizada por etapa (média, min)",
        )
        f.update_layout(xaxis_title=None)
        st.plotly_chart(f, use_container_width=True)
        st.caption(
            "Onde a barra azul passa da clara, a etapa costuma demorar mais do que o planejado."
        )

    st.subheader("Qual recurso mais pesa no prazo?")
    st.caption(
        "Simula +30 min em todas as etapas de um recurso e mostra quanto sobe a chance média de atraso."
    )
    s, base_p = sens(d, agora, risco, min(sims, 1000))
    f = px.bar(
        s.head(10).iloc[::-1],
        x="aumento_pp",
        y="recurso",
        orientation="h",
        color_discrete_sequence=[LARANJA],
    )
    f.update_layout(
        xaxis_title=f"aumento da chance de atraso (pontos percentuais) · hoje: {base_p:.0%}",
        yaxis_title=None,
    )
    st.plotly_chart(f, use_container_width=True)
    if not rec.empty:
        with st.expander("Tabela detalhada"):
            st.dataframe(rec, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 3. e se...?
with abas[2]:
    st.subheader("E se uma etapa atrasar?")
    a, b, c = st.columns(3)
    op_w = a.selectbox("Ordem", list(ativas.op) or list(r.op))
    plano = d[d.op == op_w].sort_values("ordem_etapa")[
        [
            "ordem_etapa",
            "etapa",
            "recurso",
            "dur_padrao_min",
            "plan_inicio",
            "plan_fim",
        ]
    ]
    nomes = plano.set_index("ordem_etapa").etapa
    ordem = b.selectbox(
        "Etapa que vai atrasar",
        list(nomes.index),
        format_func=lambda i: nomes[i],
    )
    atraso = c.slider("Atraso (min)", 0, 480, 120, 15)
    rp = reprogramar(plano, ordem, atraso)
    if atraso:
        st.error(
            f"A conclusão passa para {fmt(rp.novo_fim.max())}: {atraso} min depois do prazo "
            f"({fmt(plano.plan_fim.max())})."
        )
        afet = rp[rp.ordem_etapa > ordem].etapa.tolist()
        if afet:
            st.markdown(
                "**Etapas seguintes afetadas:** " + ", ".join(afet)
            )
    else:
        st.success("Sem atraso: nada muda.")
    st.caption(
        "Considera que a etapa seguinte só começa quando a anterior termina e que o tempo perdido não é recuperado."
    )
    with st.expander("Ver horários antes e depois"):
        st.dataframe(
            rp[
                [
                    "etapa",
                    "recurso",
                    "plan_inicio",
                    "novo_inicio",
                    "plan_fim",
                    "novo_fim",
                    "deslocamento_min",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("E se as condições mudarem?")
    st.caption(
        "Compare como a chance de atraso das ordens em andamento muda em cada situação."
    )
    recursos = sorted(d.recurso.unique())
    cc = st.columns(4)
    mult = cc[0].slider(
        "Tempos mais variáveis (×)",
        0.5,
        3.0,
        1.5,
        0.1,
        help="1 = variação normal. 2 = os tempos oscilam o dobro.",
    )
    rec_sel = cc[1].selectbox(
        "Recurso afetado",
        recursos,
        index=recursos.index("Tanque Hidrólise")
        if "Tanque Hidrólise" in recursos
        else 0,
    )
    extra = cc[2].slider("Tempo extra no recurso (min)", 0, 240, 60, 15)
    prazo_d = cc[3].slider("Mudança no prazo (min)", -240, 240, -120, 30)
    cen = {
        "Situação atual": {},
        "Tempos mais variáveis": dict(mult_cv=mult),
        f"+{extra} min em {rec_sel}": dict(extra_min=extra, recurso=rec_sel),
        f"Prazo {prazo_d:+d} min": dict(prazo_min=prazo_d),
        "Tudo junto": dict(
            mult_cv=mult, extra_min=extra, recurso=rec_sel, prazo_min=prazo_d
        ),
    }
    res = (
        pd.DataFrame(
            {
                n: an.avaliar(d, agora, risco, min(sims, 1000), **p)
                for n, p in cen.items()
            }
        )
        .T.reset_index(names="cenário")
    )
    f = px.bar(
        res,
        x="cenário",
        y="prob_media",
        title="Chance média de atraso em cada situação",
        color_discrete_sequence=[AZUL],
    )
    f.update_layout(
        yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None
    )
    st.plotly_chart(f, use_container_width=True)
    pior = res.iloc[1:].sort_values("prob_media").iloc[-1]
    st.info(
        f"O que mais piora o prazo: **{pior['cenário']}**, com {pior.prob_media:.0%} de chance média de "
        f"atraso (situação atual: {res.iloc[0].prob_media:.0%})."
    )
    with st.expander("Ver tabela"):
        st.dataframe(res, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 4. acompanhar uma ordem
with abas[3]:
    op = st.selectbox(
        "Escolha a ordem",
        r.sort_values("prob_atraso", ascending=False).op,
        format_func=lambda o: f"{o} · {r.set_index('op').loc[o, 'produto']}",
    )
    linha = r.set_index("op").loc[op]
    st.progress(
        int(linha.etapas_concluidas) / int(linha.total_etapas),
        text=f"{linha.etapas_concluidas} de {linha.total_etapas} etapas concluídas · {linha.status}",
    )
    g = d[d.op == op]
    prazo = g.prazo.iloc[0]
    linhas = []
    for _, e in g.iterrows():
        linhas.append(
            dict(
                etapa=e.etapa,
                ini=e.plan_inicio,
                fim=e.plan_fim,
                tipo="Planejado",
            )
        )
        if e.real_inicio <= agora:
            linhas.append(
                dict(
                    etapa=e.etapa,
                    ini=e.real_inicio,
                    fim=min(e.real_fim, agora),
                    tipo="Realizado",
                )
            )
    f = px.timeline(
        pd.DataFrame(linhas),
        x_start="ini",
        x_end="fim",
        y="etapa",
        color="tipo",
        color_discrete_map={"Planejado": "#9DB8D9", "Realizado": AZUL},
        title="Planejado × realizado",
        category_orders={"etapa": list(g.etapa)},
    )
    f.update_yaxes(autorange="reversed", title=None)
    for x, nome, cor in [(agora, "Agora", "#555"), (prazo, "Prazo", "#C62828")]:
        f.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=cor, dash="dash"),
        )
        f.add_annotation(
            x=x, y=1.04, yref="paper", text=nome, showarrow=False
        )
    st.plotly_chart(f, use_container_width=True)
    st.caption(
        "Barras claras: horários-limite do planejamento. Barras azuis: o que já foi executado até agora."
    )

    t0, mins = distribuicao_mc(d, agora, op, n_sim=sims)
    if t0 is None:
        st.info("Ordem já concluída. Não há o que simular.")
    else:
        fim_s = t0 + pd.to_timedelta(mins, unit="m")
        p50, p90 = fim_s.quantile(0.5), fim_s.quantile(0.9)

        # Explicação direta da Simulação Monte Carlo
        st.info("""
        💡 **Como ler o Monte Carlo nesta ordem:**
        - **P50 (Conclusão mais provável):** Metade das simulações conclui antes deste horário.
        - **P90 (Quase certo terminar até):** 90% das simulações concluem até aqui. Se este horário passar da linha vermelha (prazo), a ordem necessita de atenção imediata do PCP.
        """)

        m = st.columns(4)
        m[0].metric("Prazo", fmt(prazo))
        m[1].metric(
            "Conclusão mais provável (P50)",
            fmt(p50),
            help="Metade dos cenários termina antes, metade depois.",
        )
        m[2].metric(
            "Quase certo terminar até (P90)",
            fmt(p90),
            help="9 em cada 10 cenários terminam até esse horário.",
        )
        m[3].metric(
            "Chance de atraso",
            f"{(fim_s > prazo).mean():.0%}",
            help="Fração dos cenários simulados que terminam depois do prazo.",
        )
        f = px.histogram(
            x=fim_s,
            nbins=40,
            title=f"{sims} cenários simulados de conclusão",
            color_discrete_sequence=[AZUL],
        )
        for x, nome, cor in [
            (prazo, "Prazo", "#C62828"),
            (p50, "Mais provável", LARANJA),
            (p90, "90%", "#555"),
        ]:
            f.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=cor, dash="dash"),
            )
            f.add_annotation(
                x=x, y=1.04, yref="paper", text=nome, showarrow=False
            )
        f.update_layout(
            xaxis_title=None, yaxis_title="nº de cenários", bargap=0.03
        )
        st.plotly_chart(f, use_container_width=True)
        st.caption(
            "Cada barra conta quantos cenários terminam naquele horário. Tudo à direita da linha vermelha é atraso."
        )

# ---------------------------------------------------------------- 5. calcular horários
with abas[4]:
    st.markdown(
        "Informe o **produto** e o **prazo de entrega**: o sistema calcula, de trás para frente, "
        "o horário-limite de cada etapa."
    )
    a, b, c, e = st.columns(4)
    prod = a.selectbox("Produto", list(ROTEIROS))
    dia = b.date_input("Data de entrega", datetime(2026, 10, 6).date())
    hora = c.time_input("Hora de entrega", datetime(2026, 10, 6, 18, 0).time())
    folga = e.number_input("Folga entre etapas (min)", 0, 120, BUFFER_MIN, 5)
    entrega = datetime.combine(dia, hora)
    pl = pd.DataFrame(agendamento_reverso(entrega, ROTEIROS[prod], folga))
    k1, k2 = st.columns(2)
    k1.metric("A produção deve começar até", fmt(pl.plan_inicio.min()))
    k2.metric(
        "Duração total",
        f"{(entrega - pl.plan_inicio.min()).total_seconds() / 3600:.1f} h",
    )
    f = px.timeline(
        pl, x_start="plan_inicio", x_end="plan_fim", y="etapa", color="recurso"
    )
    f.update_yaxes(autorange="reversed", title=None)
    st.plotly_chart(f, use_container_width=True)
    with st.expander("Ver horários em tabela"):
        st.dataframe(
            pl[
                [
                    "ordem_etapa",
                    "etapa",
                    "recurso",
                    "dur_padrao_min",
                    "plan_inicio",
                    "plan_fim",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

# ---------------------------------------------------------------- 6. como funciona e dados
with abas[5]:
    st.markdown(f"""
**Como ler os alertas**

- 🟢 **No prazo**: a ordem deve terminar com folga de {risco} min ou mais.
- 🟡 **Atenção**: deve terminar antes do prazo, mas com menos de {risco} min de folga.
- 🔴 **Atraso previsto**: pelo andamento atual, termina depois do prazo.
- Em cada **etapa**: até {tol} min de atraso é 🟡; acima disso, 🔴.

**Como os números são calculados**

- **Horário previsto de conclusão** = tempo restante das etapas + {BUFFER_MIN} min de folga entre elas.
- **Chance de atraso** (Monte Carlo): o sistema simula a ordem {sims} vezes. Em cada vez, o tempo de cada etapa restante varia aleatoriamente em torno do padrão (distribuição lognormal, que não gera tempos negativos e permite atrasos grandes). A chance de atraso é a fração de vezes que passa do prazo.
- **Conclusão mais provável / quase certo terminar até**: a mediana (P50) e o 90º percentil (P90) dos cenários.
- A variação de cada etapa é dada pelo coeficiente de variação (CV) da tabela abaixo. As etapas são consideradas independentes entre si.
- **Todos os tempos e CVs são premissas acadêmicas**, não medições de uma fábrica real.
""")
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
    st.subheader("Dados")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button(
        "Baixar dados (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="dados_reversemilk.csv",
        mime="text/csv",
    )

rodape()