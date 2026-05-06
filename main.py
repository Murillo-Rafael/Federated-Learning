# main.py
import numpy as np
import matplotlib.pyplot as plt
from dados import gerar_dados_clientes
from cliente import Cliente
from servidor import ServidorFederado
from pipeline import ler_log, preparar_dados_log, analisar_clientes

# ==================== HIPERPARÂMETROS ====================
NUM_CLIENTES = 5
AMOSTRAS_POR_CLIENTE = 200
DIMENSAO_ENTRADA = 2
RODADAS = 30                    
EPOCAS_LOCAIS = 8
TAXA_APRENDIZADO = 0.008
# =====================================================

print("Começando!\n")

# Gerar dados
dados_clientes = gerar_dados_clientes(
    num_clientes=NUM_CLIENTES,
    amostras_por_cliente=AMOSTRAS_POR_CLIENTE,
    dimensao_entrada=DIMENSAO_ENTRADA
)

# Criar clientes
clientes = [Cliente(i, X, y) for i, (X, y) in enumerate(dados_clientes)]

# Criar servidor
servidor = ServidorFederado(dimensao_entrada=DIMENSAO_ENTRADA)

print(f"{NUM_CLIENTES} clientes criados | {RODADAS} rodadas\n")

# ===================== TREINAMENTO FEDERADO =====================
for rodada in range(1, RODADAS + 1):
    pesos_globais, vies_global = servidor.distribuir_para_clientes()
    
    atualizacoes = []
    tamanhos = []
    
    for cliente in clientes:
        pesos, vies, tamanho = cliente.treinar(pesos_globais, vies_global, 
                                              TAXA_APRENDIZADO, EPOCAS_LOCAIS)
        atualizacoes.append((pesos, vies))
        tamanhos.append(tamanho)
        
        mse_treino, mse_teste = cliente.avaliar(servidor.modelo_global)
        servidor.log(rodada, cliente.id_cliente, mse_treino, mse_teste)
    
    servidor.agregar(atualizacoes, tamanhos)
    
    if rodada % 5 == 0:
        print(f"Rodada {rodada}/{RODADAS} concluída")

print("\n Finalizado!\n")

# ===================== ANÁLISE E GRÁFICO =====================
logs = ler_log()

# Resultado da última rodada
print("Última Rodada:")
ultima = [log for log in logs if log["rodada"] == RODADAS]
for log in ultima:
    print(f"   Cliente {log['cliente_id']}: MSE = {log['mse_teste']:.4f}")

# Gráfico de Evolução
# Gráfico de Evolução
plt.figure(figsize=(10, 6))

for cliente_id in range(NUM_CLIENTES):
    cliente_data = [log for log in logs if log["cliente_id"] == cliente_id]
    rodadas = [log["rodada"] for log in cliente_data]
    mse = [log["mse_teste"] for log in cliente_data]
    label = f'Cliente {cliente_id}'
    if cliente_id == 0: label += " (Melhor caso)"
    if cliente_id == 1: label += " (Pior caso)"
    plt.plot(rodadas, mse, label=label)

plt.title('Evolução do MSE por Cliente ao Longo das Rodadas')
plt.xlabel('Rodadas')
plt.ylabel('MSE (Erro Quadrático Médio)')
plt.legend()
plt.grid(True)

# Salvar o gráfico como imagem
plt.savefig('evolucao_mse.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'evolucao_mse.png'")

plt.show()

# Comparação 80%/20%
fase1, fase2 = preparar_dados_log(logs, 0.8)

print(f"\nComparação Fase 1 (80%) vs Fase 2 (20%):")
clientes_logs = analisar_clientes(logs)

for cliente_id in [0, 1]:
    fase1_c = [x for x in fase1 if x["cliente_id"] == cliente_id]
    fase2_c = [x for x in fase2 if x["cliente_id"] == cliente_id]
    
    mse_f1 = np.mean([x['mse_teste'] for x in fase1_c]) if fase1_c else 0
    mse_f2 = np.mean([x['mse_teste'] for x in fase2_c]) if fase2_c else 0
    
    print(f"\nCliente {cliente_id} {'(Melhor caso)' if cliente_id == 0 else '(Pior caso)'}")
    print(f"   Fase 1 (80%) → MSE médio = {mse_f1:.4f}")
    print(f"   Fase 2 (20%)  → MSE médio = {mse_f2:.4f}")
    if mse_f1 > 0:
        print(f"   Melhora = {((mse_f1 - mse_f2)/mse_f1*100):.1f}%")