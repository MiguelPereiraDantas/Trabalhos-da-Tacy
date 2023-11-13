# Importar as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Criar dados de exemplo para regressão linear simples
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Característica única para cada ponto de dado
y = 4 + 3 * X + np.random.randn(100, 1)  # Relação linear com ruído gaussiano

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar um modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo usando o erro quadrático médio
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio: {mse:.2f}')

# Visualizar os resultados
plt.scatter(X_test, y_test, color='black', label='Dados de teste')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regressão Linear')
plt.xlabel('Característica')
plt.ylabel('Rótulo')
plt.legend()
plt.show()
