from http.server import BaseHTTPRequestHandler
import json
import joblib
import os

# Load model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model",
    "CNN.pkl"
)

model = joblib.load(MODEL_PATH)


class handler(BaseHTTPRequestHandler):

    def send_json(self, data, status_code=200):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

        self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_json({}, 200)

    def do_GET(self):
        self.send_json({
            "message": "Titanic Prediction API is running",
            "endpoint": "POST /api/predict"
        })

    def do_POST(self):
        try:
            # Get request body
            content_length = int(self.headers.get("Content-Length", 0))

            if content_length == 0:
                self.send_json({
                    "error": "Request body is empty"
                }, 400)
                return

            body = self.rfile.read(content_length)

            # Convert JSON to Python dictionary
            data = json.loads(body.decode("utf-8"))

            # Required fields
            required_fields = [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked"
            ]

            # Check missing fields
            missing_fields = [
                field for field in required_fields
                if field not in data
            ]

            if missing_fields:
                self.send_json({
                    "error": "Missing required fields",
                    "missing": missing_fields
                }, 400)
                return

            # Prepare features
            features = [[
                float(data["Pclass"]),
                float(data["Sex"]),
                float(data["Age"]),
                float(data["SibSp"]),
                float(data["Parch"]),
                float(data["Fare"]),
                float(data["Embarked"])
            ]]

            # Prediction
            prediction = model.predict(features)[0]

            prediction = int(prediction)

            result = (
                "Survived"
                if prediction == 1
                else "Did not survive"
            )

            # Response
            self.send_json({
                "success": True,
                "prediction": prediction,
                "result": result
            })

        except json.JSONDecodeError:
            self.send_json({
                "error": "Invalid JSON request"
            }, 400)

        except ValueError as e:
            self.send_json({
                "error": "Invalid input values",
                "details": str(e)
            }, 400)

        except Exception as e:
            self.send_json({
                "error": "Prediction failed",
                "details": str(e)
            }, 500)
