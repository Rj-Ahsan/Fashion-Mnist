from http.server import BaseHTTPRequestHandler
import json
import joblib

model = joblib.load("model/CNN.pkl")


class handler(BaseHTTPRequestHandler):

    def do_GET(self):

        response = {
            "message": "Titanic Prediction API is running",
            "endpoint": "POST /api/predict"
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        self.wfile.write(json.dumps(response).encode())

    def do_POST(self):

        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        data = json.loads(body)

        features = [[
            data["Pclass"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Fare"],
            data["Embarked"]
        ]]

        prediction = model.predict(features)[0]

        result = "Survived" if prediction == 1 else "Did not survive"

        response = {
            "prediction": int(prediction),
            "result": result
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        self.wfile.write(json.dumps(response).encode())from http.server import BaseHTTPRequestHandler
import json
import joblib

model = joblib.load("model/decision_tree.pkl")


class handler(BaseHTTPRequestHandler):

    def do_GET(self):

        response = {
            "message": "Titanic Prediction API is running",
            "endpoint": "POST /api/predict"
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        self.wfile.write(json.dumps(response).encode())

    def do_POST(self):

        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        data = json.loads(body)

        features = [[
            data["Pclass"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Fare"],
            data["Embarked"]
        ]]

        prediction = model.predict(features)[0]

        result = "Survived" if prediction == 1 else "Did not survive"

        response = {
            "prediction": int(prediction),
            "result": result
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        self.wfile.write(json.dumps(response).encode())
