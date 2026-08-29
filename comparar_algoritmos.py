# comparar_algoritmos.py
"""
Roda o pipeline federado várias vezes (múltiplas sementes) para FedAvg e
FedProx, e compara o desempenho médio +/- desvio padrão por cliente e por
grupo de heterogeneidade (clientes 0-5 vs. 6-9, que têm ruído crescente
por construção em dados.py).

Gera:
  - resultados_comparacao.csv   (uma linha por execução/cliente)
  - resumo_comparacao.csv       (média/desvio agregados por algoritmo)
  - comparacao_fedavg_fedprox.png
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from experimento import rodar_pipeline

SEMENTES = [1, 2, 3, 4, 5, 6, 7, 8]  # aumente para mais robustez estatística
ALGORITMOS = [
    {"nome": "fedavg", "mu": 0.0},
    {"nome": "fedprox", "mu": 0.01},
    {"nome": "fedprox", "mu": 0.05},
]

NUM_CLIENTES = 10
GRUPO_BAIXA_HETEROGENEIDADE = set(range(0, 6))   # clientes 0-5
GRUPO_ALTA_HETEROGENEIDADE = set(range(6, 10))   # clientes 6-9


def rotulo(cfg):
    return cfg["nome"] if cfg["nome"] == "fedavg" else f"fedprox (mu={cfg['mu']})"


def main():
    linhas = []

    for cfg in ALGORITMOS:
        for semente in SEMENTES:
            print(f"Rodando {rotulo(cfg)} | seed={semente} ...")
            resultado = rodar_pipeline(
                algoritmo=cfg["nome"],
                mu=cfg["mu"],
                semente=semente,
                personalizar=True,
                caminho_log=f"log_tmp_{cfg['nome']}_{cfg['mu']}_{semente}.csv",
                verbose=False
            )
            for cid in range(NUM_CLIENTES):
                linhas.append({
                    "algoritmo": rotulo(cfg),
                    "semente": semente,
                    "cliente_id": cid,
                    "grupo": "baixa_heterog" if cid in GRUPO_BAIXA_HETEROGENEIDADE else "alta_heterog",
                    "mse_global": resultado["mse_teste_global"][cid],
                    "mse_personalizado": resultado["mse_teste_personalizado"][cid],
                })

    # Salva resultados brutos
    with open("resultados_comparacao.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=linhas[0].keys())
        writer.writeheader()
        writer.writerows(linhas)

    # Agrega por algoritmo e grupo
    resumo = {}
    for linha in linhas:
        chave = (linha["algoritmo"], linha["grupo"])
        resumo.setdefault(chave, {"global": [], "personalizado": []})
        resumo[chave]["global"].append(linha["mse_global"])
        resumo[chave]["personalizado"].append(linha["mse_personalizado"])

    print("\n" + "=" * 78)
    print(f"{'Algoritmo':<20}{'Grupo':<16}{'MSE Global':<20}{'MSE Personalizado':<20}")
    print("=" * 78)

    resumo_linhas = []
    for (algoritmo, grupo), valores in sorted(resumo.items()):
        g_mean, g_std = np.mean(valores["global"]), np.std(valores["global"])
        p_mean, p_std = np.mean(valores["personalizado"]), np.std(valores["personalizado"])
        print(f"{algoritmo:<20}{grupo:<16}{g_mean:.3f} ± {g_std:.3f}      {p_mean:.3f} ± {p_std:.3f}")
        resumo_linhas.append({
            "algoritmo": algoritmo, "grupo": grupo,
            "mse_global_media": g_mean, "mse_global_desvio": g_std,
            "mse_personalizado_media": p_mean, "mse_personalizado_desvio": p_std,
        })

    with open("resumo_comparacao.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resumo_linhas[0].keys())
        writer.writeheader()
        writer.writerows(resumo_linhas)

    print("\nSalvo: resultados_comparacao.csv (bruto) e resumo_comparacao.csv (agregado)")

    # Gráfico: MSE global médio por algoritmo, separado por grupo
    algoritmos_unicos = sorted(set(l["algoritmo"] for l in linhas))
    grupos = ["baixa_heterog", "alta_heterog"]

    x = np.arange(len(algoritmos_unicos))
    largura = 0.35

    plt.figure(figsize=(10, 6))
    for i, grupo in enumerate(grupos):
        medias = [np.mean(resumo[(alg, grupo)]["global"]) for alg in algoritmos_unicos]
        offset = (i - 0.5) * largura
        plt.bar(x + offset, medias, largura, label=f"Grupo {grupo}")

    plt.xticks(x, algoritmos_unicos, rotation=15)
    plt.ylabel("MSE de Teste Global (média entre seeds)")
    plt.title("FedAvg vs. FedProx por grupo de heterogeneidade")
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig("comparacao_fedavg_fedprox.png", dpi=300, bbox_inches="tight")
    print("Gráfico salvo como 'comparacao_fedavg_fedprox.png'")


if __name__ == "__main__":
    main()
