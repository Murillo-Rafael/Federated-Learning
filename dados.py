# dados.py
import numpy as np

def gerar_dados_clientes(num_clientes: int = 5, 
                        amostras_por_cliente: int = 200,
                        dimensao_entrada: int = 2, 
                        semente: int = 42):
    """Gera dados simulados não-IID (heterogêneos) para cada cliente."""
    
    np.random.seed(semente)
    dados_clientes = []
    
    pesos_verdadeiros = np.array([2.5, -1.8])
    vies_verdadeiro = 10.0

    for i in range(num_clientes):
        # Criar heterogeneidade (não-IID) entre os clientes
        ruido_pesos = 0.1 + (i * 0.15)      # Aumenta gradualmente
        ruido_vies = 0.1 + (i * 0.3)
        
        pesos_cliente = pesos_verdadeiros + np.random.randn(dimensao_entrada) * ruido_pesos
        vies_cliente = vies_verdadeiro + np.random.randn() * ruido_vies

        # Gerar features
        X = np.random.randn(amostras_por_cliente, dimensao_entrada) * 2
        
        # Gerar targets com ruído
        y = X @ pesos_cliente + vies_cliente + np.random.randn(amostras_por_cliente) * 1.2

        dados_clientes.append((X, y))
    
    return dados_clientes