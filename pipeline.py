# pipeline.py
import csv
import numpy as np
from collections import defaultdict


import csv
import numpy as np
from collections import defaultdict


def ler_log(caminho_log='log_clientes.csv'):
    logs = []

    try:
        with open(caminho_log, mode='r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                logs.append({
                    "janela": int(row["janela"]),
                    "rodada": int(row["rodada"]),
                    "cliente_id": int(row["cliente_id"]),
                    "selecionado": int(row["selecionado"]),
                    "falhou": int(row["falhou"]),
                    "tempo_execucao": float(row["tempo_execucao"]),
                    "tamanho_lote": int(row["tamanho_lote"]),
                    "mse_treino": float(row["mse_treino"]),
                    "mse_teste": float(row["mse_teste"])
                })

    except FileNotFoundError:
        print(f"Arquivo {caminho_log} não encontrado.")
        return []

    return logs


def analisar_clientes(logs):
    clientes = defaultdict(list)

    for log in logs:
        clientes[log["cliente_id"]].append(log)

    return dict(clientes)


def dividir_log(logs, proporcao_treino=0.7):
    """Divide os logs em treino e teste de forma aleatória."""
    if not logs:
        return [], []
    
    logs_shuffle = logs.copy()
    np.random.shuffle(logs_shuffle)
    
    corte = int(len(logs_shuffle) * proporcao_treino)
    
    treino = logs_shuffle[:corte]
    teste = logs_shuffle[corte:]
    
    return treino, teste


def preparar_dados_log(logs, proporcao_primeira_fase=0.8):
    "Divide os logs de forma cronológica (por rodada)."
    if not logs:
        return [], []
    
    # Ordena por rodada para manter sequência temporal
    logs_ordenados = sorted(logs, key=lambda x: x["rodada"])
    
    corte = int(len(logs_ordenados) * proporcao_primeira_fase)
    
    fase1 = logs_ordenados[:corte]      # Primeiras rodadas
    fase2 = logs_ordenados[corte:]      # Últimas rodadas
    
    return fase1, fase2


def analisar_clientes(logs):
    """Agrupa os logs por cliente."""
    clientes = defaultdict(list)
    
    for log in logs:
        clientes[log["cliente_id"]].append(log)
    
    return dict(clientes)