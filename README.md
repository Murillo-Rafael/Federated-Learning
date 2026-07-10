# Federated Learning — Simulação com Janelas Temporais

Simulação didática de **Aprendizado Federado (Federated Learning)** implementada em Python puro (NumPy), sem frameworks externos de FL. O projeto reproduz um cenário realista de treinamento distribuído: clientes com dados heterogêneos (non-IID), indisponibilidade/falhas aleatórias, timeout de execução e treinamento incremental por janelas temporais.

## Como funciona

O servidor central mantém um modelo global de **regressão linear**. A cada rodada:

1. O servidor distribui os parâmetros globais (pesos e viés) para os clientes.
2. Um subconjunto de clientes é sorteado para participar da rodada.
3. Cada cliente sorteado treina localmente sobre uma **janela deslizante** de seus próprios dados, simulando dados que chegam ao longo do tempo.
4. Clientes podem **falhar** (por indisponibilidade simulada) ou **estourar o tempo limite**, sendo descartados da rodada.
5. O servidor agrega as atualizações recebidas por **média ponderada pelo tamanho da amostra** (estilo FedAvg).
6. Tudo é registrado em `log_clientes.csv` para análise posterior.

Ao final, o script gera um gráfico (`evolucao_mse_treino.png`) mostrando a evolução do MSE de treino por cliente.

## Estrutura do projeto

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Orquestra o pipeline federado: rodadas, janelas, treino e avaliação final |
| `dados.py` | Gera dados sintéticos por cliente, com ruído variável (dados non-IID) |
| `modelo.py` | Modelo de regressão linear (predição, MSE, gradiente descendente) |
| `cliente.py` | Classe `Cliente`: treino local sobre janelas de dados |
| `servidor.py` | Classe `ServidorFederado`: agregação de parâmetros e log em CSV |
| `pipeline.py` | Funções auxiliares para ler e analisar o log de execução |

## Parâmetros principais (`main.py`)

| Parâmetro | Descrição |
|---|---|
| `NUM_CLIENTES` | Número total de clientes simulados |
| `MINIMO_AMOSTRAS` / `MAXIMO_AMOSTRAS` | Faixa de volume de dados por cliente |
| `RODADAS_POR_JANELA` | Rodadas de comunicação por janela temporal |
| `EPOCAS_LOCAIS` | Épocas de treino local por rodada |
| `TAMANHO_JANELA` / `SALTO_JANELA` | Configuração da janela deslizante sobre os dados |
| `CLIENTES_POR_RODADA` | Quantos clientes são sorteados por rodada |
| `PROBABILIDADE_FALHA` | Chance de um cliente selecionado falhar |
| `TEMPO_LIMITE` | Timeout (em segundos) para o treino local de um cliente |

## Como executar

```bash
pip install -r requirements.txt
python main.py
```

Isso vai:
- Gerar os dados sintéticos dos clientes
- Rodar o treinamento federado por todas as janelas temporais
- Salvar o log de cada rodada em `log_clientes.csv`
- Avaliar o modelo final com os dados de teste reservados de cada cliente
- Salvar o gráfico de evolução do MSE em `evolucao_mse_treino.png`

## Limitações conhecidas

- O modelo é uma regressão linear simples — adequado para fins didáticos, não para tarefas complexas.
- Não há mecanismos de privacidade (ex: differential privacy, criptografia homomórfica) — o foco é a lógica de coordenação federada, não a preservação de privacidade.
- Os dados são sintéticos, gerados artificialmente para simular heterogeneidade entre clientes.

## Possíveis extensões

- Trocar a regressão linear por um modelo mais expressivo (ex: MLP)
- Adicionar privacidade diferencial na agregação
- Persistir o histórico entre execuções para comparar diferentes configurações de hiperparâmetros
