# Importar as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Criar dados de exemplo para classificação binária
np.random.seed(42)
X = np.random.rand(100, 2)  # Duas características para cada ponto de dado
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Classificação binária baseada na soma das características

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar um modelo de Regressão Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

# Exibir o relatório de classificação
print('\nRelatório de Classificação:')
print(classification_report(y_test, y_pred))