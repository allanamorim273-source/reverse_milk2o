"""Análises do ReverseMilk v2: gargalos, sensibilidade por recurso e cenários what-if."""
import pandas as pd

from agendamento import BUFFER_MIN
from indicadores import resumo_ops


def minutos_perdidos(d):
    """Minutos perdidos por etapa já concluída = estouro do tempo padrão + espera acima do buffer.
    (Só conta o que já aconteceu até o instante atual.)"""
    c = d[d.situacao == "Concluída"].copy()
    c["perda_ciclo_min"] = c.desvio_ciclo_min.clip(lower=0)
    c["perda_espera_min"] = (c.espera_min - BUFFER_MIN).clip(lower=0).fillna(0)
    c["perda_total_min"] = c.perda_ciclo_min + c.perda_espera_min
    c["estourou"] = c.desvio_ciclo_min > 0.05 * c.dur_padrao_min  # >5% acima do padrão
    return c


def ranking(d, por):
    """Ranking de gargalos agrupado por 'recurso' ou 'etapa'."""
    c = minutos_perdidos(d)
    g = c.groupby(por).agg(
        execucoes=("op", "count"),
        perda_total_min=("perda_total_min", "sum"),
        perda_media_min=("perda_total_min", "mean"),
        pct_estouro=("estourou", "mean"),
        variabilidade_cv=("cv", "mean"),
    ).reset_index()
    total = g.perda_total_min.sum()
    g["pct_da_perda"] = g.perda_total_min / total if total else 0.0
    return g.sort_values("perda_total_min", ascending=False).reset_index(drop=True)


def aplicar(d, mult_cv=1.0, extra_min=0.0, recurso=None, prazo_min=0.0):
    """Altera só o que ainda não aconteceu (etapas não concluídas) e os prazos."""
    x = d.copy()
    pend = x.situacao != "Concluída"
    x.loc[pend, "cv"] = x.loc[pend, "cv"] * mult_cv
    if recurso and extra_min:
        x.loc[pend & (x.recurso == recurso), "dur_padrao_min"] += extra_min
    x["prazo"] = x["prazo"] + pd.Timedelta(minutes=prazo_min)
    return x


def avaliar(d, agora, limite_risco, sims, **cenario):
    """Resumo de um cenário sobre as ordens ainda ativas."""
    r = resumo_ops(aplicar(d, **cenario), agora, limite_risco, n_sim=sims)
    a = r[~r.status.str.startswith("Concluída")]
    return dict(
        ativas=len(a),
        prob_media=float(a.prob_atraso.mean()) if len(a) else 0.0,
        atrasadas=int((a.status == "Atrasada").sum()),
        em_risco=int((a.status == "Em risco").sum()),
        margem_media_min=float(a.margem_min.mean()) if len(a) else 0.0,
    )


def sensibilidade(d, agora, limite_risco, sims, extra_min=30):
    """Quanto a probabilidade média de atraso sobe se cada recurso demorar +extra_min por etapa."""
    base = avaliar(d, agora, limite_risco, sims)["prob_media"]
    linhas = []
    for rec in sorted(d.recurso.unique()):
        p = avaliar(d, agora, limite_risco, sims, extra_min=extra_min, recurso=rec)["prob_media"]
        linhas.append(dict(recurso=rec, aumento_pp=(p - base) * 100))
    return pd.DataFrame(linhas).sort_values("aumento_pp", ascending=False).reset_index(drop=True), base


def conclusoes(d, r):
    """Frases-resumo geradas automaticamente a partir dos dados."""
    f = []
    a = r[~r.status.str.startswith("Concluída")]
    if len(a):
        n = int((a.status != "No prazo").sum())
        f.append(f"**{n} de {len(a)}** ordens ativas estão em risco ou com atraso projetado.")
    rec = ranking(d, "recurso")
    if len(rec):
        t = rec.iloc[0]
        f.append(f"**{t.recurso}** concentra {t.pct_da_perda:.0%} dos minutos perdidos até agora (recurso mais crítico).")
        e = ranking(d, "etapa").iloc[0]
        f.append(f"A etapa que mais perde tempo é **{e.etapa}**, que estourou o tempo padrão "
                 f"em {e.pct_estouro:.0%} das execuções.")
    return f or ["Ainda não há dados suficientes. Avance o momento atual na barra lateral."]


def causa_principal(d, r):
    """Para cada ordem atrasada, o recurso que mais tempo fez perder; conta quantas ordens por recurso."""
    ops = r[r.status.isin(["Atrasada", "Concluída com atraso"])].op
    c = minutos_perdidos(d)
    c = c[c.op.isin(ops) & (c.perda_total_min > 0)]
    if c.empty:
        return pd.DataFrame(columns=["recurso", "ordens"])
    top = c.loc[c.groupby("op").perda_total_min.idxmax()]
    return top.recurso.value_counts().rename_axis("recurso").reset_index(name="ordens")


def atraso_por_posicao(d):
    """Por posição na rota: perda média gerada na etapa e atraso médio acumulado no fim dela."""
    c = minutos_perdidos(d)
    return c.groupby("ordem_etapa").agg(perda_media=("perda_total_min", "mean"),
                                        atraso_medio=("atraso_min", "mean")).reset_index()


def planejado_vs_real(d):
    """Duração padrão (planejada) x duração real média por etapa, só etapas concluídas."""
    c = d[d.situacao == "Concluída"].copy()
    c["realizado"] = c.dur_padrao_min + c.desvio_ciclo_min
    return c.groupby("etapa").agg(planejado=("dur_padrao_min", "mean"),
                                  realizado=("realizado", "mean")).reset_index()
