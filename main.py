# main.py

import numpy as np
import matplotlib.pyplot as plt
from dados import gerar_dados_clientes
from cliente import Cliente
from servidor import ServidorFederado
from pipeline import ler_log

# HIPERPARÂMETROS
NUM_CLIENTES = 10
MINIMO_AMOSTRAS = 200
MAXIMO_AMOSTRAS = 800
DIMENSAO_ENTRADA = 2

RODADAS_POR_JANELA = 10
EPOCAS_LOCAIS = 8
TAXA_APRENDIZADO = 0.008

TAMANHO_JANELA = 80
SALTO_JANELA = 40

CLIENTES_POR_RODADA = 6
PROBABILIDADE_FALHA = 0.25
TEMPO_LIMITE = 0.05

SEMENTE = 42
np.random.seed(SEMENTE)

print("Começando pipeline federado com janelas temporais!\n")

# Gerar dados com volumes diferentes
dados_clientes = gerar_dados_clientes(
    num_clientes=NUM_CLIENTES,
    minimo_amostras=MINIMO_AMOSTRAS,
    maximo_amostras=MAXIMO_AMOSTRAS,
    dimensao_entrada=DIMENSAO_ENTRADA,
    semente=SEMENTE
)

# Criar clientes
clientes = [Cliente(i, X, y) for i, (X, y) in enumerate(dados_clientes)]

# Criar servidor
servidor = ServidorFederado(dimensao_entrada=DIMENSAO_ENTRADA)

# Descobrir número máximo de janelas possível
maior_treino = max(len(cliente.y_treino) for cliente in clientes)
NUM_JANELAS = max(1, int((maior_treino - TAMANHO_JANELA) / SALTO_JANELA) + 1)

print(f"{NUM_CLIENTES} clientes criados")
print(f"Clientes por rodada: {CLIENTES_POR_RODADA}")
print(f"Probabilidade de falha: {PROBABILIDADE_FALHA}")
print(f"Tempo limite por cliente: {TEMPO_LIMITE}s")
print(f"Janelas temporais: {NUM_JANELAS}")
print(f"Tamanho da janela: {TAMANHO_JANELA}")
print(f"Salto da janela: {SALTO_JANELA}\n")

# TREINAMENTO FEDERADO COM JANELAS
for janela in range(NUM_JANELAS):

    print(f"\n=== Janela {janela + 1}/{NUM_JANELAS} ===")

    for rodada in range(1, RODADAS_POR_JANELA + 1):

        pesos_globais, vies_global = servidor.distribuir_para_clientes()

        atualizacoes = []
        tamanhos = []

        clientes_ids = np.arange(NUM_CLIENTES)
        clientes_sorteados = np.random.choice(
            clientes_ids,
            size=min(CLIENTES_POR_RODADA, NUM_CLIENTES),
            replace=False
        )

        for cliente in clientes:

            selecionado = cliente.id_cliente in clientes_sorteados
            falhou = False
            tempo_execucao = 0.0
            tamanho_lote = 0
            mse_treino = 0.0
            mse_teste = 0.0

            if selecionado:

                # Simula falha/intermitência
                if np.random.rand() < PROBABILIDADE_FALHA:
                    falhou = True

                else:
                    pesos, vies, tamanho_lote, tempo_execucao = cliente.treinar(
                        pesos_globais,
                        vies_global,
                        TAXA_APRENDIZADO,
                        EPOCAS_LOCAIS,
                        janela,
                        TAMANHO_JANELA,
                        SALTO_JANELA
                    )

                    # Simula tempo limite
                    if tempo_execucao > TEMPO_LIMITE:
                        falhou = True

                    elif pesos is not None and tamanho_lote > 0:
                        atualizacoes.append((pesos, vies))
                        tamanhos.append(tamanho_lote)

            # Loga todos os clientes, inclusive não selecionados ou falhos
            servidor.log(
                janela + 1,
                rodada,
                cliente.id_cliente,
                selecionado,
                falhou,
                tempo_execucao,
                tamanho_lote,
                mse_treino,
                mse_teste
            )

        agregou = servidor.agregar(atualizacoes, tamanhos)

        # Após agregação, avalia apenas clientes participantes válidos
        for cliente in clientes:
            if cliente.id_cliente in clientes_sorteados and agregou:
                mse_treino = cliente.avaliar_treino_janela(
                    servidor.modelo_global,
                    janela,
                    TAMANHO_JANELA,
                    SALTO_JANELA
                )

                servidor.log(
                    janela + 1,
                    rodada,
                    cliente.id_cliente,
                    True,
                    False,
                    0.0,
                    0,
                    mse_treino,
                    0.0
                )

        if rodada % 5 == 0:
            print(f"Rodada {rodada}/{RODADAS_POR_JANELA} concluída")

print("\nTreinamento finalizado!\n")

# TESTE FINAL
print("Teste final com os dados reservados:\n")

for cliente in clientes:
    mse_teste = cliente.avaliar_teste(servidor.modelo_global)

    print(f"Cliente {cliente.id_cliente} | MSE Teste Final = {mse_teste:.4f}")

    servidor.log(
        NUM_JANELAS,
        RODADAS_POR_JANELA,
        cliente.id_cliente,
        True,
        False,
        0.0,
        len(cliente.y_teste),
        0.0,
        mse_teste
    )

# ANÁLISE DO LOG
logs = ler_log()

plt.figure(figsize=(10, 6))

for cliente_id in range(NUM_CLIENTES):
    dados_cliente = [
        log for log in logs
        if log["cliente_id"] == cliente_id
        and log["mse_treino"] > 0
        and log["falhou"] == 0
        and log["selecionado"] == 1
    ]

    if not dados_cliente:
        continue

    dados_cliente = sorted(
        dados_cliente,
        key=lambda x: (x["janela"], x["rodada"])
    )

    eixo_x = list(range(1, len(dados_cliente) + 1))
    mse_treino = [log["mse_treino"] for log in dados_cliente]

    plt.plot(eixo_x, mse_treino, label=f"Cliente {cliente_id}")

plt.title("Evolução do MSE de Treino por Cliente")
plt.xlabel("Atualizações válidas registradas")
plt.ylabel("MSE de Treino")
plt.legend()
plt.grid(True)

plt.savefig("evolucao_mse_treino.png", dpi=300, bbox_inches="tight")
print("\nGráfico salvo como 'evolucao_mse_treino.png'")

plt.show()