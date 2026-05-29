import numpy as np

def gerar_dados_clientes(
    num_clientes=10,
    minimo_amostras=200,
    maximo_amostras=800,
    dimensao_entrada=2,
    semente=42
):
    np.random.seed(semente)
    dados_clientes = []

    pesos_verdadeiros = np.array([2.5, -1.8])
    vies_verdadeiro = 10.0

    for i in range(num_clientes):
        qtd_amostras = np.random.randint(minimo_amostras, maximo_amostras + 1)

        ruido_pesos = 0.1 + (i * 0.08)
        ruido_vies = 0.1 + (i * 0.15)

        pesos_cliente = pesos_verdadeiros + np.random.randn(dimensao_entrada) * ruido_pesos
        vies_cliente = vies_verdadeiro + np.random.randn() * ruido_vies

        X = np.random.randn(qtd_amostras, dimensao_entrada) * 2
        y = X @ pesos_cliente + vies_cliente + np.random.randn(qtd_amostras) * 1.2

        dados_clientes.append((X, y))

    return dados_clientes