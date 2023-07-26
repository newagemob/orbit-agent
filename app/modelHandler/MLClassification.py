from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path

app_path = Path(__file__).parent.parent


class ActionClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)
        self.model.eval()

    def preprocess(self, text):
        inputs = self.tokenizer([text], return_tensors="pt")
        return inputs

    def predict(self, text):
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            return predicted_class.item()


if __name__ == "__main__":
    model_path = f"{app_path}/modelHandler/output/classificationModels/combined"
    classifier = ActionClassifier(model_path)
    test_prediction = "2022-08-15,288.2140462802,291.3635945834,287.3325523865,290.6604003906,18085700.0,4.3461953096895005,21772860.0"
    predicted_class = classifier.predict(test_prediction)
    print(predicted_class)
