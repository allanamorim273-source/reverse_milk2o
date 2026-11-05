"""Gera dados sintéticos (simulação estocástica) de ordens de produção."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from agendamento import BUFFER_MIN, agendamento_reverso, ruido_lognormal

# (etapa, recurso, duração padrão em min, coeficiente de variação)
ROTEIROS = {
    "Leite Zero Lactose": [
        ("Liberação da matéria-prima", "Recepção", 60, 0.10),
        ("Pasteurização", "Pasteurizador 1", 90, 0.08),
        ("Transferência", "Tanque Pulmão", 45, 0.15),
        ("Hidrólise enzimática", "Tanque Hidrólise", 480, 0.12),
        ("Formulação", "Tanque Formulação", 120, 0.10),
        ("Liberação da qualidade", "Laboratório", 240, 0.25),
    ],
    "Leite UHT": [
        ("Liberação da matéria-prima", "Recepção", 60, 0.10),
        ("Pasteurização", "Pasteurizador 1", 90, 0.08),
        ("Padronização", "Tanque Padronização", 60, 0.10),
        ("Tratamento UHT", "UHT 1", 120, 0.08),
        ("Envase", "Linha Envase 1", 180, 0.15),
        ("Liberação da qualidade", "Laboratório", 180, 0.25),
    ],
    "Queijo": [
        ("Liberação da matéria-prima", "Recepção", 60, 0.10),
        ("Pasteurização", "Pasteurizador 2", 90, 0.08),
        ("Coagulação e corte", "Tanque Queijo", 120, 0.12),
        ("Prensagem", "Prensa 1", 240, 0.15),
        ("Salga", "Tanque Salmoura", 480, 0.10),
        ("Liberação da qualidade", "Laboratório", 240, 0.25),
    ],
    "Leite em Pó": [
        ("Liberação da matéria-prima", "Recepção", 60, 0.10),
        ("Pasteurização", "Pasteurizador 2", 90, 0.08),
        ("Evaporação", "Evaporador 1", 180, 0.12),
        ("Secagem (spray dryer)", "Torre de Secagem", 240, 0.10),
        ("Embalagem", "Linha Embalagem 2", 120, 0.15),
        ("Liberação da qualidade", "Laboratório", 240, 0.25),
    ],
    "Leite Condensado": [
        ("Liberação da matéria-prima", "Recepção", 60, 0.10),
        ("Pasteurização", "Pasteurizador 1", 90, 0.08),
        ("Concentração", "Evaporador 2", 150, 0.12),
        ("Envase", "Linha Envase 3", 150, 0.15),
        ("Liberação da qualidade", "Laboratório", 180, 0.25),
    ],
}


def gerar(n_ops=40, seed=42, base=None):
    """Cria n_ops ordens com plano (agendamento reverso) e execução simulada."""
    rng = np.random.default_rng(seed)
    base = base or datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    produtos = list(ROTEIROS)
    linhas = []
    for i in range(n_ops):
        produto = produtos[rng.integers(len(produtos))]
        etapas = ROTEIROS[produto]
        total = sum(e[2] for e in etapas) + BUFFER_MIN * (len(etapas) - 1)
        prazo = base + timedelta(minutes=int(total + rng.uniform(0, 36 * 60)))
        plano = agendamento_reverso(prazo, etapas)

        # execução simulada: começa (às vezes) atrasada e sofre ruído + paradas
        t = plano[0]["plan_inicio"]
        if rng.random() < 0.3:
            t += timedelta(minutes=float(rng.exponential(25)))
        for p in plano:
            dur = p["dur_padrao_min"] * float(ruido_lognormal(rng, p["cv"]))
            if rng.random() < 0.03:  # parada/ocorrência
                dur += float(rng.uniform(30, 120))
            real_fim = t + timedelta(minutes=dur)
            linhas.append({**p, "op": f"OP-{1000 + i}", "produto": produto,
                           "prazo": prazo, "real_inicio": t, "real_fim": real_fim})
            t = real_fim + timedelta(minutes=float(rng.exponential(7)))
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    df = gerar(base=datetime(2026, 10, 5, 6, 0))
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/dados_sinteticos.csv", index=False)
    print(f"{len(df)} linhas salvas em data/dados_sinteticos.csv")
