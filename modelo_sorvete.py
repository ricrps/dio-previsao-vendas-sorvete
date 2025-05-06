import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Carregar dados
df = pd.read_csv('inputs/dados.csv')
X = df[['temperatura']]
y = df['vendas']

# Iniciar experimento MLflow
mlflow.start_run()

# Modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Previsão
y_pred = modelo.predict(X)

# Avaliação
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'RMSE: {rmse:.2f}')

# Logar no MLflow
mlflow.log_param("modelo", "LinearRegression")
mlflow.log_metric("rmse", rmse)
mlflow.sklearn.log_model(modelo, "modelo_sorvete")

mlflow.end_run()
