import joblib

model = joblib.load("models/intent_classifier.pkl")

def predict_intent(prompt: str):
    intent = model.predict([prompt])[0]
    return intent

if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    intent = predict_intent(prompt)
    print(f"Predicted intent: {intent}")
