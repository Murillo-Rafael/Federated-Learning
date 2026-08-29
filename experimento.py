# experimento.py
"""
Núcleo reutilizável do pipeline federado.

Extraído de main.py para permitir rodar o mesmo experimento com diferentes
sementes e algoritmos de agregação (FedAvg / FedProx) sem duplicar código,
o que é necessário para gerar comparações estatisticamente mais confiáveis
para o relatório (múltiplas execuções em vez de uma única rodada).
"""

import numpy as np
from dados import gerar_dados_clientes
from cliente import Cliente
from servidor import ServidorFederado


def rodar_pipeline(
    algoritmo="fedavg",           # "fedavg" ou "fedprox"
    mu=0.01,                      # coeficiente proximal (usado só se algoritmo="fedprox")
    semente=42,
    num_clientes=10,
    minimo_amostras=200,
    maximo_amostras=800,
    dimensao_entrada=2,
    rodadas_por_janela=10,
    epocas_locais=8,
    taxa_aprendizado=0.008,
    tamanho_janela=80,
    salto_janela=40,
    clientes_por_rodada=6,
    probabilidade_falha=0.25,
    tempo_limite=0.05,
    personalizar=True,
    epocas_personalizacao=30,
    taxa_personalizacao=0.01,
    caminho_log=None,
    verbose=False
):
    """Executa o pipeline federado completo e retorna um dicionário com os
    resultados finais (MSE de teste globais e, opcionalmente, personalizados
    por cliente), além do modelo global e da lista de clientes."""

    np.random.seed(semente)

    dados_clientes = gerar_dados_clientes(
        num_clientes=num_clientes,
        minimo_amostras=minimo_amostras,
        maximo_amostras=maximo_amostras,
        dimensao_entrada=dimensao_entrada,
        semente=semente
    )

    clientes = [Cliente(i, X, y) for i, (X, y) in enumerate(dados_clientes)]

    servidor = ServidorFederado(
        dimensao_entrada=dimensao_entrada,
        caminho_log=caminho_log or f"log_clientes_{algoritmo}_seed{semente}.csv"
    )

    maior_treino = max(len(cliente.y_treino) for cliente in clientes)
    num_janelas = max(1, int((maior_treino - tamanho_janela) / salto_janela) + 1)

    if verbose:
        print(f"[{algoritmo} | seed={semente}] {num_clientes} clientes, "
              f"{num_janelas} janelas, falha={probabilidade_falha}, mu={mu if algoritmo == 'fedprox' else '-'}")

    for janela in range(num_janelas):
        for rodada in range(1, rodadas_por_janela + 1):

            pesos_globais, vies_global = servidor.distribuir_para_clientes()

            atualizacoes = []
            tamanhos = []

            clientes_ids = np.arange(num_clientes)
            clientes_sorteados = np.random.choice(
                clientes_ids,
                size=min(clientes_por_rodada, num_clientes),
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
                    if np.random.rand() < probabilidade_falha:
                        falhou = True
                    else:
                        pesos, vies, tamanho_lote, tempo_execucao = cliente.treinar(
                            pesos_globais,
                            vies_global,
                            taxa_aprendizado,
                            epocas_locais,
                            janela,
                            tamanho_janela,
                            salto_janela,
                            algoritmo=algoritmo,
                            mu=mu
                        )

                        if tempo_execucao > tempo_limite:
                            falhou = True
                        elif pesos is not None and tamanho_lote > 0:
                            atualizacoes.append((pesos, vies))
                            tamanhos.append(tamanho_lote)

                servidor.log(
                    janela + 1, rodada, cliente.id_cliente, selecionado, falhou,
                    tempo_execucao, tamanho_lote, mse_treino, mse_teste
                )

            agregou = servidor.agregar(atualizacoes, tamanhos)

            for cliente in clientes:
                if cliente.id_cliente in clientes_sorteados and agregou:
                    mse_treino = cliente.avaliar_treino_janela(
                        servidor.modelo_global, janela, tamanho_janela, salto_janela
                    )
                    servidor.log(
                        janela + 1, rodada, cliente.id_cliente, True, False,
                        0.0, 0, mse_treino, 0.0
                    )

    # Avaliação final (modelo global, sem personalização)
    pesos_globais_finais, vies_global_final = servidor.modelo_global.obter_parametros()

    mse_teste_global = {}
    mse_teste_personalizado = {}

    for cliente in clientes:
        mse_teste_global[cliente.id_cliente] = cliente.avaliar_teste(servidor.modelo_global)

        if personalizar:
            pesos_p, vies_p = cliente.personalizar(
                pesos_globais_finais, vies_global_final,
                taxa_personalizacao, epocas_personalizacao
            )
            mse_teste_personalizado[cliente.id_cliente] = cliente.avaliar_teste_personalizado(pesos_p, vies_p)

    if verbose:
        for cid in range(num_clientes):
            linha = f"  Cliente {cid} | MSE global = {mse_teste_global[cid]:.4f}"
            if personalizar:
                linha += f" | MSE personalizado = {mse_teste_personalizado[cid]:.4f}"
            print(linha)

    return {
        "algoritmo": algoritmo,
        "mu": mu,
        "semente": semente,
        "mse_teste_global": mse_teste_global,
        "mse_teste_personalizado": mse_teste_personalizado if personalizar else None,
        "clientes": clientes,
        "servidor": servidor,
        "num_janelas": num_janelas
    }
