# main.py

import matplotlib.pyplot as plt
from pipeline import ler_log
from experimento import rodar_pipeline

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

# Algoritmo de agregação: "fedavg" ou "fedprox"
ALGORITMO = "fedavg"
MU = 0.01  # só usado se ALGORITMO == "fedprox"

# Personalização: fine-tuning local a partir do modelo global final
PERSONALIZAR = True
EPOCAS_PERSONALIZACAO = 30
TAXA_PERSONALIZACAO = 0.01

print("Começando pipeline federado com janelas temporais!\n")
print(f"Algoritmo: {ALGORITMO}" + (f" (mu={MU})" if ALGORITMO == "fedprox" else ""))

resultado = rodar_pipeline(
    algoritmo=ALGORITMO,
    mu=MU,
    semente=SEMENTE,
    num_clientes=NUM_CLIENTES,
    minimo_amostras=MINIMO_AMOSTRAS,
    maximo_amostras=MAXIMO_AMOSTRAS,
    dimensao_entrada=DIMENSAO_ENTRADA,
    rodadas_por_janela=RODADAS_POR_JANELA,
    epocas_locais=EPOCAS_LOCAIS,
    taxa_aprendizado=TAXA_APRENDIZADO,
    tamanho_janela=TAMANHO_JANELA,
    salto_janela=SALTO_JANELA,
    clientes_por_rodada=CLIENTES_POR_RODADA,
    probabilidade_falha=PROBABILIDADE_FALHA,
    tempo_limite=TEMPO_LIMITE,
    personalizar=PERSONALIZAR,
    epocas_personalizacao=EPOCAS_PERSONALIZACAO,
    taxa_personalizacao=TAXA_PERSONALIZACAO,
    caminho_log="log_clientes.csv",
    verbose=False
)

print(f"\nTreinamento finalizado! ({resultado['num_janelas']} janelas)\n")
print("Teste final com os dados reservados:\n")

mse_global = resultado["mse_teste_global"]
mse_pers = resultado["mse_teste_personalizado"]

for cid in range(NUM_CLIENTES):
    linha = f"Cliente {cid} | MSE Global = {mse_global[cid]:.4f}"
    if PERSONALIZAR:
        ganho = mse_global[cid] - mse_pers[cid]
        linha += f" | MSE Personalizado = {mse_pers[cid]:.4f} | Ganho = {ganho:+.4f}"
    print(linha)

if PERSONALIZAR:
    media_global = sum(mse_global.values()) / NUM_CLIENTES
    media_pers = sum(mse_pers.values()) / NUM_CLIENTES
    print(f"\nMédia MSE Global      : {media_global:.4f}")
    print(f"Média MSE Personalizado: {media_pers:.4f}")
    print(f"Redução média         : {(1 - media_pers / media_global) * 100:.1f}%")

# ANÁLISE DO LOG
logs = ler_log("log_clientes.csv")

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

    dados_cliente = sorted(dados_cliente, key=lambda x: (x["janela"], x["rodada"]))

    eixo_x = list(range(1, len(dados_cliente) + 1))
    mse_treino = [log["mse_treino"] for log in dados_cliente]

    plt.plot(eixo_x, mse_treino, label=f"Cliente {cliente_id}")

plt.title(f"Evolução do MSE de Treino por Cliente ({ALGORITMO.upper()})")
plt.xlabel("Atualizações válidas registradas")
plt.ylabel("MSE de Treino")
plt.legend()
plt.grid(True)

plt.savefig("evolucao_mse_treino.png", dpi=300, bbox_inches="tight")
print("\nGráfico salvo como 'evolucao_mse_treino.png'")

if PERSONALIZAR:
    plt.figure(figsize=(10, 6))
    ids = list(range(NUM_CLIENTES))
    largura = 0.35
    plt.bar([i - largura/2 for i in ids], [mse_global[i] for i in ids], largura, label="Global")
    plt.bar([i + largura/2 for i in ids], [mse_pers[i] for i in ids], largura, label="Personalizado")
    plt.xlabel("Cliente")
    plt.ylabel("MSE de Teste Final")
    plt.title("MSE de Teste: Modelo Global vs. Personalizado")
    plt.xticks(ids)
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig("mse_global_vs_personalizado.png", dpi=300, bbox_inches="tight")
    print("Gráfico salvo como 'mse_global_vs_personalizado.png'")

plt.show()
