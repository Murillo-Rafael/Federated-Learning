from modelo import ModeloRegressaoLinear
import numpy as np
import time


class Cliente:
    def __init__(self, id_cliente, X, y, proporcao_treino=0.8):
        self.id_cliente = id_cliente

        corte = int(len(X) * proporcao_treino)

        self.X_treino = X[:corte]
        self.y_treino = y[:corte]

        self.X_teste = X[corte:]
        self.y_teste = y[corte:]

    def obter_janela(self, indice_janela, tamanho_janela, salto):
        inicio = indice_janela * salto
        fim = inicio + tamanho_janela

        if inicio >= len(self.y_treino):
            return None, None

        return self.X_treino[inicio:fim], self.y_treino[inicio:fim]

    def treinar(
        self,
        pesos_globais,
        vies_global,
        taxa_aprendizado,
        epocas_locais,
        indice_janela,
        tamanho_janela,
        salto,
        algoritmo="fedavg",
        mu=0.0
    ):
        X_janela, y_janela = self.obter_janela(
            indice_janela,
            tamanho_janela,
            salto
        )

        if X_janela is None or len(y_janela) == 0:
            return None, None, 0, 0.0

        inicio_tempo = time.time()

        modelo = ModeloRegressaoLinear(len(pesos_globais))
        modelo.definir_parametros(pesos_globais, vies_global)

        mu_efetivo = mu if algoritmo == "fedprox" else 0.0

        modelo.treinar_local(
            X_janela,
            y_janela,
            taxa_aprendizado,
            epocas_locais,
            mu=mu_efetivo,
            pesos_referencia=pesos_globais,
            vies_referencia=vies_global
        )

        tempo_execucao = time.time() - inicio_tempo

        pesos, vies = modelo.obter_parametros()

        return pesos, vies, len(y_janela), tempo_execucao

    def personalizar(self, pesos_globais, vies_global, taxa_aprendizado, epocas_locais):
        """Fine-tuning local a partir do modelo global final.

        Parte dos parâmetros globais e dá alguns passos extras de treino
        usando todo o histórico de treino local do cliente. Isso costuma
        reduzir bastante o erro de clientes cujos dados divergem mais da
        distribuição média usada para o modelo global (non-IID).
        """
        if len(self.y_treino) == 0:
            return pesos_globais, vies_global

        modelo = ModeloRegressaoLinear(len(pesos_globais))
        modelo.definir_parametros(pesos_globais, vies_global)

        modelo.treinar_local(
            self.X_treino,
            self.y_treino,
            taxa_aprendizado,
            epocas_locais
        )

        return modelo.obter_parametros()

    def avaliar_teste_personalizado(self, pesos, vies):
        modelo = ModeloRegressaoLinear(len(pesos))
        modelo.definir_parametros(pesos, vies)
        return modelo.erro_quadratico_medio(self.X_teste, self.y_teste)

    def avaliar_treino_janela(self, modelo, indice_janela, tamanho_janela, salto):
        X_janela, y_janela = self.obter_janela(
            indice_janela,
            tamanho_janela,
            salto
        )

        if X_janela is None or len(y_janela) == 0:
            return 0.0

        return modelo.erro_quadratico_medio(X_janela, y_janela)

    def avaliar_teste(self, modelo):
        return modelo.erro_quadratico_medio(self.X_teste, self.y_teste)