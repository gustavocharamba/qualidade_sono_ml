import pandas as pd
import joblib

from preprocessing import preprocessing

from models.linear_regression_model import linear_regression
from models.decision_tree_model import decision_tree
from models.random_forest_model import random_forest

df = pd.read_csv('data/sleep_and_health.csv')

df = preprocessing(df)

data_encoded = pd.get_dummies(df, columns=['Occupation', 'Gender', 'BMI'], drop_first=True)

y = data_encoded['Quality']
X = data_encoded.drop('Quality', axis=1)

linear = linear_regression(X, y)
decision = decision_tree(X, y)
forest = random_forest(X, y)

print("\n===== Resultados Regressão Linear =====\n")
print(linear.describe())

print("\n===== Resultados Árvore de Decisão =====\n")
print(decision.describe())

print("\n===== Resultados Random Forest Regression =====\n")
print(forest.describe())

print("\n ====== Comparação ===== \n")

# Lista para organizar os resultados
lista_modelos = [
    ('Regressão Linear', linear),
    ('Árvore de Decisão', decision),
    ('Random Forest', forest)
]

dados_comparativos = []

for nome, df_resultados in lista_modelos:
    #Médias de Performance
    r2_medio = df_resultados['R2_teste'].mean()
    mae_medio = df_resultados['MAE_teste'].mean()
    rmse_medio = df_resultados['RMSE_teste'].mean()

    # Desvio Padrão
    # Quanto menor, melhor. Indica que o modelo não depende de sorte.
    desvio_padrao = df_resultados['R2_teste'].std()

    # Overfitting
    # Diferença entre Treino e Teste. Se for alto, o modelo decorou.
    overfitting = df_resultados['R2_treino'].mean() - r2_medio

    dados_comparativos.append({
        'Modelo': nome,
        'R2 (Qualidade)': r2_medio,
        'MAE (Erro Absoluto)': mae_medio,
        'RMSE (Erro Quad.)': rmse_medio,
        'Desvio Padrão': desvio_padrao,  # Mantido conforme solicitado
        'Overfitting': overfitting
    })

# Criar DataFrame da comparação
df_ranking = pd.DataFrame(dados_comparativos)

# Ordenar pelo melhor R2 (Maior é melhor)
df_ranking = df_ranking.sort_values(by='R2 (Qualidade)', ascending=False)

# Exibir formatado
print(df_ranking.round(4).to_string(index=False))

# Identifica o melhor
melhor_nome = df_ranking.iloc[0]['Modelo']
mae_vencedor = df_ranking.iloc[0]['MAE (Erro Absoluto)']
print(f"\nMelhor modelo: {melhor_nome}")

model_to_save = None

# Configurações otimizadas para o modelo final
if melhor_nome == 'Regressão Linear':
    from sklearn.linear_model import LinearRegression

    model_to_save = LinearRegression()

elif melhor_nome == 'Árvore de Decisão':
    from sklearn.tree import DecisionTreeRegressor

    # Limitando profundidade para evitar overfitting no deploy
    model_to_save = DecisionTreeRegressor(max_depth=10, random_state=42)

elif melhor_nome == 'Random Forest':
    from sklearn.ensemble import RandomForestRegressor

    model_to_save = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Treina com 100% dos dados (X e y) e salva o arquivo
if model_to_save:
    model_to_save.fit(X, y)

    filename = 'melhor_modelo_sono.pkl'
    joblib.dump(model_to_save, filename)

else:
    print("Erro ao selecionar o modelo.")