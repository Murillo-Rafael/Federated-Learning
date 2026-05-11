# cliente.py

from modelo import ModeloRegressaoLinear
import numpy as np


class Cliente:
    "Cliente federado"

    def __init__(self, id_cliente, X, y, proporcao_treino=0.8):
        self.id_cliente = id_cliente

        # Embaralha os dados
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Divide treino e teste
        corte = int(len(X) * proporcao_treino)

        self.X_treino = X[:corte]
        self.y_treino = y[:corte]

        self.X_teste = X[corte:]
        self.y_teste = y[corte:]

    def obter_lote_rodada(self, rodada, total_rodadas):
        """
        Retorna apenas uma parte do treino para cada rodada
        """

        total = len(self.y_treino)

        inicio = int((rodada - 1) * total / total_rodadas)
        fim = int(rodada * total / total_rodadas)

        if fim <= inicio:
            fim = min(inicio + 1, total)

        return (
            self.X_treino[inicio:fim],
            self.y_treino[inicio:fim]
        )

    def treinar(
        self,
        pesos_globais,
        vies_global,
        taxa_aprendizado,
        epocas_locais,
        rodada,
        total_rodadas
    ):

        # Pega apenas o lote da rodada
        X_lote, y_lote = self.obter_lote_rodada(
            rodada,
            total_rodadas
        )

        # Cria modelo local
        modelo = ModeloRegressaoLinear(len(pesos_globais))

        # Recebe modelo global
        modelo.definir_parametros(
            pesos_globais,
            vies_global
        )

        # Treina localmente
        modelo.treinar_local(
            X_lote,
            y_lote,
            taxa_aprendizado,
            epocas_locais
        )

        pesos, vies = modelo.obter_parametros()

        return pesos, vies, len(y_lote)

    def avaliar_treino(
        self,
        modelo,
        rodada,
        total_rodadas
    ):

        total = len(self.y_treino)

        fim = int(rodada * total / total_rodadas)

        X = self.X_treino[:fim]
        y = self.y_treino[:fim]

        return modelo.erro_quadratico_medio(X, y)

    def avaliar_teste(self, modelo):

        return modelo.erro_quadratico_medio(
            self.X_teste,
            self.y_teste
        )