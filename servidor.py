import csv
import numpy as np
from modelo import ModeloRegressaoLinear


class ServidorFederado:
    def __init__(self, dimensao_entrada: int, caminho_log='log_clientes.csv'):
        self.modelo_global = ModeloRegressaoLinear(dimensao_entrada)
        self.caminho_log = caminho_log

        with open(self.caminho_log, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "janela",
                "rodada",
                "cliente_id",
                "selecionado",
                "falhou",
                "tempo_execucao",
                "tamanho_lote",
                "mse_treino",
                "mse_teste"
            ])

    def distribuir_para_clientes(self):
        return self.modelo_global.obter_parametros()

    def agregar(self, atualizacoes, tamanhos):
        if len(atualizacoes) == 0:
            return False

        total = sum(tamanhos)

        if total == 0:
            return False

        novos_pesos = np.zeros_like(atualizacoes[0][0])
        novo_vies = 0.0

        for (pesos, vies), tamanho in zip(atualizacoes, tamanhos):
            peso = tamanho / total
            novos_pesos += peso * pesos
            novo_vies += peso * vies

        self.modelo_global.definir_parametros(novos_pesos, novo_vies)
        return True

    def log(
        self,
        janela,
        rodada,
        cliente_id,
        selecionado,
        falhou,
        tempo_execucao,
        tamanho_lote,
        mse_treino,
        mse_teste
    ):
        with open(self.caminho_log, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                janela,
                rodada,
                cliente_id,
                int(selecionado),
                int(falhou),
                round(float(tempo_execucao), 6),
                int(tamanho_lote),
                round(float(mse_treino), 6),
                round(float(mse_teste), 6)
            ])