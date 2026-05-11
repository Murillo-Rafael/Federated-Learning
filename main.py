# main.py

import numpy as np
import matplotlib.pyplot as plt
from dados import gerar_dados_clientes
from cliente import Cliente
from servidor import ServidorFederado
from pipeline import ler_log

# HIPERPARÂMETROS
NUM_CLIENTES = 5
AMOSTRAS_POR_CLIENTE = 200
DIMENSAO_ENTRADA = 2
RODADAS = 30
EPOCAS_LOCAIS = 8
TAXA_APRENDIZADO = 0.008

print("Começando!\n")

# Gerar dados
dados_clientes = gerar_dados_clientes(
    num_clientes=NUM_CLIENTES,
    amostras_por_cliente=AMOSTRAS_POR_CLIENTE,
    dimensao_entrada=DIMENSAO_ENTRADA
)

# Criar clientes
clientes = [Cliente(i, X, y) for i, (X, y) in enumerate(dados_clientes)]

# Criar servidor
servidor = ServidorFederado(dimensao_entrada=DIMENSAO_ENTRADA)

print(f"{NUM_CLIENTES} clientes criados | {RODADAS} rodadas\n")

# TREINAMENTO FEDERADO
for rodada in range(1, RODADAS + 1):

    pesos_globais, vies_global = servidor.distribuir_para_clientes()

    atualizacoes = []
    tamanhos = []

    # Treinamento local com lote proporcional à rodada
    for cliente in clientes:
        pesos, vies, tamanho = cliente.treinar(
            pesos_globais,
            vies_global,
            TAXA_APRENDIZADO,
            EPOCAS_LOCAIS,
            rodada,
            RODADAS
        )

        atualizacoes.append((pesos, vies))
        tamanhos.append(tamanho)

    # Agregação FedAvg
    servidor.agregar(atualizacoes, tamanhos)

    # Avaliação após atualizar o modelo global
    for cliente in clientes:
        mse_treino = cliente.avaliar_treino(
            servidor.modelo_global,
            rodada,
            RODADAS
        )

        # Teste apenas na última rodada usando os 20% separados
        if rodada == RODADAS:
            mse_teste = cliente.avaliar_teste(servidor.modelo_global)
        else:
            mse_teste = 0.0

        servidor.log(
            rodada,
            cliente.id_cliente,
            mse_treino,
            mse_teste
        )

    if rodada % 5 == 0:
        print(f"Rodada {rodada}/{RODADAS} concluída")

print("\nFinalizado!\n")

# ANÁLISE
logs = ler_log()

# Resultado da última rodada
print("Última Rodada - Teste Final:")
ultima = [log for log in logs if log["rodada"] == RODADAS]

for log in ultima:
    print(f"   Cliente {log['cliente_id']}: MSE Teste = {log['mse_teste']:.4f}")

# GRÁFICO DO TREINO
plt.figure(figsize=(10, 6))

for cliente_id in range(NUM_CLIENTES):
    cliente_data = [log for log in logs if log["cliente_id"] == cliente_id]

    rodadas = [log["rodada"] for log in cliente_data]
    mse_treino = [log["mse_treino"] for log in cliente_data]

    label = f"Cliente {cliente_id}"
    if cliente_id == 0:
        label += " (Melhor caso)"
    if cliente_id == 1:
        label += " (Pior caso)"

    plt.plot(rodadas, mse_treino, label=label)

plt.title("Evolução do MSE de Treino por Cliente")
plt.xlabel("Rodadas")
plt.ylabel("MSE de Treino")
plt.legend()
plt.grid(True)

plt.savefig("evolucao_mse_treino.png", dpi=300, bbox_inches="tight")
print("Gráfico salvo como 'evolucao_mse_treino.png'")

plt.show()

# RESUMO FINAL DO TESTE
print("\nResumo do Teste Final com os 20% reservados:")

for log in ultima:
    print(
        f"Cliente {log['cliente_id']} | "
        f"MSE Teste Final = {log['mse_teste']:.4f}"
    )