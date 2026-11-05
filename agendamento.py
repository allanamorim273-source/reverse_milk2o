"""Núcleo do ReverseMilk: agendamento reverso (backward scheduling)."""
from datetime import timedelta

import numpy as np
import pandas as pd

BUFFER_MIN = 15  # folga planejada entre etapas (minutos)


def ruido_lognormal(rng, cv, size=None):
    """Fator multiplicativo com média 1 e coeficiente de variação cv."""
    sigma = np.sqrt(np.log(1 + cv**2))
    return rng.lognormal(-sigma**2 / 2, sigma, size)


def agendamento_reverso(prazo, etapas, buffer_min=BUFFER_MIN):
    """Parte do prazo de entrega e calcula, de trás para frente, o horário-limite
    de início e fim de cada etapa.

    etapas: lista de (nome, recurso, duracao_min, cv) NA ORDEM do processo.
    """
    resultado = []
    fim = prazo
    for ordem in range(len(etapas) - 1, -1, -1):
        nome, recurso, dur, cv = etapas[ordem]
        inicio = fim - timedelta(minutes=dur)
        resultado.append(dict(
            ordem_etapa=ordem + 1, etapa=nome, recurso=recurso,
            dur_padrao_min=dur, cv=cv, plan_inicio=inicio, plan_fim=fim,
        ))
        fim = inicio - timedelta(minutes=buffer_min)
    return resultado[::-1]


def reprogramar(plano, ordem_atrasada, atraso_min):
    """Se a etapa `ordem_atrasada` terminar `atraso_min` minutos depois do plano,
    desloca as etapas seguintes e devolve o novo cronograma."""
    p = plano.sort_values("ordem_etapa").copy()
    delta = pd.Timedelta(minutes=atraso_min)
    p["novo_inicio"] = p.plan_inicio.where(p.ordem_etapa <= ordem_atrasada, p.plan_inicio + delta)
    p["novo_fim"] = p.plan_fim.where(p.ordem_etapa < ordem_atrasada, p.plan_fim + delta)
    p["deslocamento_min"] = (p.novo_fim - p.plan_fim).dt.total_seconds() / 60
    return p
