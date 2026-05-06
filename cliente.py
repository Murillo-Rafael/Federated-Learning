# cliente.py
from modelo import ModeloRegressaoLinear
import numpy as np

class Cliente:
    """Representa um cliente com divisão treino/teste."""
    
    def __init__(self, id_cliente: int, X: np.ndarray, y: np.ndarray, proporcao_treino=0.7):
        self.id_cliente = id_cliente

        # Vai embaralhando aq
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]

        corte = int(len(X) * proporcao_treino)

        # Aq divide treino e teste
        self.X_treino = X[:corte]
        self.y_treino = y[:corte]
        self.X_teste = X[corte:]
        self.y_teste = y[corte:]

    def treinar(self, pesos_globais, vies_global, taxa_aprendizado, epocas_locais):
        "Treina o modelo local a partir dos parâmetros globais."
        modelo_local = ModeloRegressaoLinear(len(pesos_globais))
        modelo_local.definir_parametros(pesos_globais, vies_global)

        modelo_local.treinar_local(
            self.X_treino,
            self.y_treino,
            taxa_aprendizado,
            epocas_locais
        )

        pesos, vies = modelo_local.obter_parametros()
        return pesos, vies, len(self.y_treino)

    def avaliar(self, modelo):
        """Avalia o modelo nos dados de treino e teste do cliente.
        Retorna (mse_treino, mse_teste)
        """
        mse_treino = modelo.erro_quadratico_medio(self.X_treino, self.y_treino)
        mse_teste = modelo.erro_quadratico_medio(self.X_teste, self.y_teste)
        return mse_treino, mse_teste