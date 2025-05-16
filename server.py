import http.server
import socketserver
import json
import urllib.parse
import os
from pathlib import Path
import joblib
import numpy as np

MODELS_PATH = Path("models")

# Cargar modelos
scaler = joblib.load(MODELS_PATH / "scaler.pkl")
logistic = joblib.load(MODELS_PATH / "logistic_regression.pkl")
svm = joblib.load(MODELS_PATH / "svm.pkl")
neural_net = joblib.load(MODELS_PATH / "neural_net.pkl")
weights = joblib.load(MODELS_PATH / "pesos_fcm.pkl")

# Función para predecir con FCM
def fcm_predict(X, weights, threshold=0.5):
    activations = X @ weights  # matriz (N,30) * (30,30)
    max_activation = np.max(activations, axis=1, keepdims=True)
    binary_output = (activations >= threshold * max_activation).astype(int)
    final_output = binary_output.max(axis=1)  # decisión final binaria
    return final_output

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = urllib.parse.parse_qs(body.decode())

            # Convertir a diccionario plano
            input_data = {k: v[0] for k, v in data.items()}

            # Obtener valores ordenados C1 a C30
            input_list = [float(input_data[f"C{i}"]) for i in range(1, 31)]
            input_array = np.array([input_list])
            input_scaled = scaler.transform(input_array)

            # Predicciones
            result = {
                "logistic": int(logistic.predict(input_scaled)[0]),
                "svm": int(svm.predict(input_scaled)[0]),
                "neural_net": int(neural_net.predict(input_scaled)[0]),
                "fcm": int(fcm_predict(input_scaled, weights)[0])
            }

            # Enviar respuesta JSON
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path == "/batch_predict":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                import cgi
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'})
                
                # Obtener el archivo CSV y modelo seleccionado
                file_item = form['csv']
                model_name = form.getvalue('model').lower()

                if file_item.file:
                    import pandas as pd
                    from sklearn.metrics import confusion_matrix, accuracy_score

                    df = pd.read_csv(file_item.file)

                    if "C31" not in df.columns:
                        self.send_error(400, "El dataset debe incluir la columna 'C31' como etiqueta.")
                        return

                    y_true = df["C31"]
                    X = df[[f"C{i}" for i in range(1, 31)]]
                    X_scaled = scaler.transform(X)

                    # Selección del modelo
                    if model_name == "logistic":
                        model = logistic
                    elif model_name == "svm":
                        model = svm
                    elif model_name == "neural_net":
                        model = neural_net
                    elif model_name == "fcm":
                        X_array = X.values
                        y_pred = fcm_predict(X_array, weights)
                        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
                        cm = confusion_matrix(y_true, y_pred).tolist()
                    else:
                        self.send_error(400, "Modelo no válido.")
                        return

                    if model_name != "fcm":
                        y_pred = model.predict(X_scaled)
                        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
                        cm = confusion_matrix(y_true, y_pred).tolist()

                    # Respuesta JSON
                    result = {
                        "accuracy": acc,
                        "confusion_matrix": cm
                    }

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                else:
                    self.send_error(400, "Archivo CSV no proporcionado")
            else:
                self.send_error(400, "Tipo de contenido no soportado")

        else:
            self.send_error(404, "Not found")

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080)) 

    print(f"Servidor iniciado en http://localhost:{PORT}")
    with socketserver.TCPServer(("0.0.0.0", PORT), MyHandler) as httpd:
        httpd.serve_forever()

