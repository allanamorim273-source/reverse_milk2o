"""Indicadores, status das etapas/ordens e simulação de Monte Carlo."""
from datetime import timedelta

import numpy as np
import pandas as pd

from agendamento import BUFFER_MIN, ruido_lognormal


def classificar_etapas(df, agora, tol_min=30):
    """Adiciona situação, atraso, desvio de ciclo e tempo de espera a cada etapa."""
    d = df.sort_values(["op", "ordem_etapa"]).copy()
    d["situacao"] = np.select(
        [d.real_fim <= agora, d.real_inicio <= agora],
        ["Concluída", "Em execução"], default="Pendente")

    atraso_fim = (d.real_fim - d.plan_fim).dt.total_seconds() / 60
    atraso_ini = (d.real_inicio - d.plan_inicio).dt.total_seconds() / 60
    atraso_agora = (agora - d.plan_fim).dt.total_seconds() / 60
    atraso_exec = np.maximum(np.maximum(atraso_ini, atraso_agora), 0)
    d["atraso_min"] = np.select(
        [d.situacao == "Concluída", d.situacao == "Em execução"],
        [atraso_fim, atraso_exec], default=0.0)

    d["status_etapa"] = np.select(
        [d.situacao == "Pendente", d.atraso_min <= 0, d.atraso_min <= tol_min],
        ["Pendente", "No prazo", "Em risco"], default="Atrasada")

    dur_real = (d.real_fim - d.real_inicio).dt.total_seconds() / 60
    d["desvio_ciclo_min"] = np.where(d.situacao == "Concluída", dur_real - d.dur_padrao_min, np.nan)
    espera = (d.real_inicio - d.groupby("op").real_fim.shift()).dt.total_seconds() / 60
    d["espera_min"] = np.where(d.situacao == "Pendente", np.nan, espera)
    return d


def _restantes(g, agora, buffer):
    """Etapas ainda não concluídas de uma OP (usa só o que é conhecido até `agora`).
    Devolve (lista de (duração, cv, já decorrido, espera antes), instante t0, etapa atual)."""
    exec_ = g[g.situacao == "Em execução"]
    pend = g[g.situacao == "Pendente"]
    rest, espera_primeira = [], 0
    if len(exec_):
        e = exec_.iloc[0]
        decorrido = (agora - e.real_inicio).total_seconds() / 60
        rest.append((e.dur_padrao_min, e.cv, decorrido, 0))
        t0, etapa_atual = agora, e.etapa
    elif (g.situacao == "Concluída").any():
        t0, etapa_atual, espera_primeira = agora, "Aguardando " + pend.iloc[0].etapa, buffer
    else:
        t0, etapa_atual = max(agora, g.plan_inicio.iloc[0]), "Não iniciada"
    for k, (_, p) in enumerate(pend.iterrows()):
        esp = espera_primeira if (k == 0 and not len(exec_)) else buffer
        rest.append((p.dur_padrao_min, p.cv, 0, esp))
    return rest, t0, etapa_atual


def _simular(rest, n_sim, rng):
    """Minutos totais restantes em n_sim cenários (Monte Carlo)."""
    total = np.zeros(n_sim)
    for dur, cv, dec, esp in rest:
        amostra = dur * ruido_lognormal(rng, cv, n_sim)
        total += np.maximum(amostra - dec, 0.1 * dur) + esp
    return total


def distribuicao_mc(d, agora, op, buffer=BUFFER_MIN, n_sim=2000, seed=7):
    """Devolve (t0, minutos simulados) de uma OP, ou (None, None) se já concluída."""
    g = d[d.op == op].sort_values("ordem_etapa")
    if g.iloc[-1].situacao == "Concluída":
        return None, None
    rest, t0, _ = _restantes(g, agora, buffer)
    return t0, _simular(rest, n_sim, np.random.default_rng(seed))


def resumo_ops(d, agora, limite_risco_min=60, buffer=BUFFER_MIN, n_sim=1000, seed=7):
    """Uma linha por ordem: projeção de conclusão, margem, prob. de atraso (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    linhas = []
    for op, g in d.groupby("op"):
        g = g.sort_values("ordem_etapa")
        prazo = g.prazo.iloc[0]
        base = dict(op=op, produto=g.produto.iloc[0], prazo=prazo,
                    etapas_concluidas=int((g.situacao == "Concluída").sum()),
                    total_etapas=len(g))
        ultima = g.iloc[-1]
        if ultima.situacao == "Concluída":
            margem = (prazo - ultima.real_fim).total_seconds() / 60
            linhas.append({**base, "etapa_atual": "—", "projecao_conclusao": ultima.real_fim,
                           "p90_conclusao": ultima.real_fim, "margem_min": margem,
                           "prob_atraso": float(margem < 0),
                           "status": "Concluída no prazo" if margem >= 0 else "Concluída com atraso"})
            continue

        rest, t0, etapa_atual = _restantes(g, agora, buffer)
        total_det = sum(max(dur - dec, 0.1 * dur) + esp for dur, cv, dec, esp in rest)
        projecao = t0 + timedelta(minutes=total_det)
        margem = (prazo - projecao).total_seconds() / 60

        total_mc = _simular(rest, n_sim, rng)
        folga = (prazo - t0).total_seconds() / 60
        prob = float((total_mc > folga).mean())
        p90 = t0 + timedelta(minutes=float(np.percentile(total_mc, 90)))

        status = "Atrasada" if margem < 0 else ("Em risco" if margem < limite_risco_min else "No prazo")
        linhas.append({**base, "etapa_atual": etapa_atual, "projecao_conclusao": projecao,
                       "p90_conclusao": p90, "margem_min": margem, "prob_atraso": prob,
                       "status": status})
    return pd.DataFrame(linhas)
