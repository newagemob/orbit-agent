import pandas as pd
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

app_path = Path(__file__).parent.parent


class FinancialMLPipeline:
    def __init__(self, classification_data_path, prediction_data_path):
        self.classification_data_path = classification_data_path
        self.prediction_data_path = prediction_data_path
        self.model = None
        self.tokenizer = None

    def load_classification_data(self):
        data = pd.read_csv(self.classification_data_path)
        features = data.drop(columns=["Timestamp", "Action"]).values
        labels = data["Action"].map({"Buy": 0, "Sell": 1, "Hold": 2}).values
        return features, labels

    def prepare_classification_data(self):
        features, labels = self.load_classification_data()
        # Split the data into train and validation sets
        train_features, val_features, train_labels, val_labels = train_test_split(
            features, labels, test_size=0.2
        )
        return train_features, train_labels, val_features, val_labels

    def create_classification_model(self, model_name="bert-base-uncased", num_labels=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def fine_tune_classification_model(
        self, train_features, train_labels, val_features, val_labels, output_dir
    ):
        train_dataset = FinancialDataset(train_features, train_labels, self.tokenizer)
        val_dataset = FinancialDataset(val_features, val_labels, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{app_path}/modelHandler/output/logs",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)

    def load_lstm_predictions(self):
        # Implement logic to load predictions from your LSTM model
        # and extract features from the prediction dataset
        lstm_predictions = None
        lstm_features = None
        return lstm_predictions, lstm_features

    def unify_datasets(
        self,
        lstm_predictions,
        lstm_features,
        classification_features,
        classification_labels,
    ):
        # Combine LSTM predictions and classification features
        if lstm_features is not None and classification_features is not None:
            unified_features = np.concatenate(
                [lstm_features, classification_features], axis=1
            )
        elif lstm_features is not None:
            unified_features = lstm_features
        elif classification_features is not None:
            unified_features = classification_features
        else:
            raise ValueError("Both LSTM features and classification features are None.")

        unified_labels = classification_labels
        return unified_features, unified_labels

    def train_pipeline(self):
        (
            train_features,
            train_labels,
            val_features,
            val_labels,
        ) = self.prepare_classification_data()

        lstm_predictions, lstm_features = self.load_lstm_predictions()
        classification_features, classification_labels = self.load_classification_data()

        unified_features, unified_labels = self.unify_datasets(
            lstm_predictions,
            lstm_features,
            classification_features,
            classification_labels,
        )

        # model name for the classification model - get ticker from before first underscore
        model_name = self.classification_data_path.split("/")[-1].split("_")[0]

        self.create_classification_model()
        self.fine_tune_classification_model(
            train_features=unified_features,
            train_labels=unified_labels,
            val_labels=val_labels,
            val_features=val_features,
            output_dir=f"{app_path}/modelHandler/output/classificationModels/{model_name}",
        )


class FinancialDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length=128):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = " ".join(map(str, self.features[idx]))
        label = self.labels[idx]

        encoding = self.tokenizer(
            feature,
            return_tensors="pt",
            padding="max_length",  # Add padding to make features equal length
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


if __name__ == "__main__":
    classification_data_path = f"{app_path}/dataHandler/output/classificationDatasets/combined_dataset.csv"
    prediction_data_path = f"{app_path}/dataHandler/output/predictionDatasets/combined_dataset.csv"

    pipeline = FinancialMLPipeline(classification_data_path, prediction_data_path)
    pipeline.train_pipeline()
