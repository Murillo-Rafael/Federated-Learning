# modelo.py
import numpy as np

class ModeloRegressaoLinear:
    "Modelo de regressão linear simples."
    
    def __init__(self, dimensao_entrada: int):
        self.pesos = np.zeros(dimensao_entrada)
        self.viés = 0.0

    def obter_parametros(self):
        "Retorna cópia dos parâmetros do modelo."
        return self.pesos.copy(), self.viés

    def definir_parametros(self, pesos: np.ndarray, viés: float):
        """Define os parâmetros do modelo."""
        self.pesos = pesos.copy()
        self.viés = float(viés)

    def prever(self, X: np.ndarray) -> np.ndarray:
        "Faz predição."
        return X @ self.pesos + self.viés

    def erro_quadratico_medio(self, X: np.ndarray, y: np.ndarray) -> float:
        "Calcula o Mean Squared Error (MSE)."
        y_pred = self.prever(X)
        return np.mean((y_pred - y) ** 2)

    def treinar_local(self, X: np.ndarray, y: np.ndarray,
                     taxa_aprendizado: float = 0.01, epocas_locais: int = 5,
                     mu: float = 0.0, pesos_referencia: np.ndarray = None,
                     vies_referencia: float = None):
        """Treinamento local usando Gradiente Descendente.

        Se mu > 0, aplica o termo proximal do FedProx: penaliza o quanto os
        parâmetros locais se afastam dos parâmetros globais recebidos no
        início da rodada (pesos_referencia / vies_referencia). Isso ajuda a
        conter a divergência de clientes com dados muito heterogêneos
        (non-IID), sem exigir nenhuma mudança no lado do servidor.
        """
        n = len(y)
        if n == 0:
            raise ValueError("Conjunto de treino tá vazio!")

        usar_proximal = mu > 0.0 and pesos_referencia is not None and vies_referencia is not None

        for _ in range(epocas_locais):
            y_pred = self.prever(X)
            erro = y_pred - y

            grad_w = (2 / n) * (X.T @ erro)
            grad_b = (2 / n) * np.sum(erro)

            if usar_proximal:
                grad_w += 2 * mu * (self.pesos - pesos_referencia)
                grad_b += 2 * mu * (self.viés - vies_referencia)

            self.pesos -= taxa_aprendizado * grad_w
            self.viés -= taxa_aprendizado * grad_b