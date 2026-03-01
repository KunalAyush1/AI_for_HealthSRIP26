import pickle
import numpy as np
import pandas as pd
import yaml
import os  

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def load_dataset(path):
    """
    Load preprocessed dataset from disk.
    """
    with open(path, "rb") as f:
        X, y, participants = pickle.load(f)
    return np.array(X), np.array(y), np.array(participants)


def normalize(X):
    """
    Normalize dataset using global mean and standard deviation.
    """
    mean = X.mean()
    std = X.std()
    return (X - mean) / (std + 1e-8)


class BreathingDataset(Dataset):
    """
    PyTorch Dataset for breathing signal windows.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].permute(1, 0), self.y[idx]


class CNN1D(nn.Module):
    """
    1D CNN model.
    """

    def __init__(self, num_classes):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self._to_linear = None
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 960)
            x = self.pool1(self.relu(self.conv1(x)))
            x = self.pool2(self.relu(self.conv2(x)))
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    """
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, loader, device):
    """
    Evaluate model.
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)


def main():

    EPOCHS = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X, y, participants = load_dataset("Dataset/dataset.pkl")
    print("Dataset shape:", X.shape)

    X = normalize(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Classes:", list(le.classes_))

    unique_participants = np.unique(participants)

    results = []

    metrics_dict = {
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        },
        "metrics": {}
    }

    
    os.makedirs("models", exist_ok=True)

    for test_p in unique_participants:

        print(f"\n===== Testing on {test_p} =====")

        train_idx = np.where(participants != test_p)[0]
        test_idx  = np.where(participants == test_p)[0]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        train_dataset = BreathingDataset(X_train, y_train)
        test_dataset = BreathingDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        model = CNN1D(num_classes=len(le.classes_)).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            loss = train_model(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        y_true, y_pred = evaluate_model(model, test_loader, device)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("Confusion Matrix:\n", cm)

        
        model_path = os.path.join("models", "cnn1.pkl")

        save_dict = {
            "model_state_dict": model.state_dict(),
            "label_classes": list(le.classes_),
            "input_shape": (960, 3)
        }

        with open(model_path, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"Model saved to: {model_path}")

        results.append({
            "participant": test_p,
            "accuracy": acc,
            "precision": prec,
            "recall": rec
        })

        metrics_dict["metrics"][str(test_p)] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec)
        }

    df = pd.DataFrame(results)

    avg_metrics = df[["accuracy", "precision", "recall"]].mean()

    metrics_dict["average"] = {
        "accuracy": float(avg_metrics["accuracy"]),
        "precision": float(avg_metrics["precision"]),
        "recall": float(avg_metrics["recall"])
    }

    with open("metrics.yaml", "w") as f:
        yaml.dump(metrics_dict, f, sort_keys=False)

    print("\nMetrics saved to metrics.yaml")


if __name__ == "__main__":
    main()