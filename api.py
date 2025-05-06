from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

# Carrega o modelo do MLflow
model = mlflow.sklearn.load_model("mlruns/0/<ID_DO_RUN>/artifacts/modelo_sorvete")  # Substitua <ID_DO_RUN> pelo valor correto

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    temperatura = data['temperatura']
    prediction = model.predict([[temperatura]])
    return jsonify({'temperatura': temperatura, 'vendas_previstas': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
