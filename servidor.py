# servidor.py
import csv
import numpy as np
from modelo import ModeloRegressaoLinear

class ServidorFederado:
    def __init__(self, dimensao_entrada: int, caminho_log='log_clientes.csv'):
        self.modelo_global = ModeloRegressaoLinear(dimensao_entrada)
        self.caminho_log = caminho_log
        
        # Criar arquivo de log com cabeçalho
        with open(self.caminho_log, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["rodada", "cliente_id", "mse_treino", "mse_teste"])

    def distribuir_para_clientes(self):
        """Retorna os parâmetros atuais do modelo global."""
        return self.modelo_global.obter_parametros()

    def agregar(self, atualizacoes, tamanhos):
        """Agregação federada (FedAvg) - média ponderada pelos tamanhos dos datasets."""
        total = sum(tamanhos)
        novos_pesos = np.zeros_like(atualizacoes[0][0])
        novo_vies = 0.0

        for (pesos, vies), tamanho in zip(atualizacoes, tamanhos):
            peso = tamanho / total
            novos_pesos += peso * pesos
            novo_vies += peso * vies

        self.modelo_global.definir_parametros(novos_pesos, novo_vies)

    def avaliar_global(self, X: np.ndarray, y: np.ndarray) -> float:
        """Avalia o modelo global em um conjunto de dados."""
        return self.modelo_global.erro_quadratico_medio(X, y)

    def log(self, rodada: int, cliente_id: int, mse_treino: float, mse_teste: float):
        """Registra os resultados de um cliente no arquivo CSV."""
        with open(self.caminho_log, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                rodada, 
                cliente_id, 
                round(float(mse_treino), 6), 
                round(float(mse_teste), 6)
            ])