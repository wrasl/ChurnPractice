import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import build_preprocessing_pipeline
from model import ChurnMLP
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

df = pd.read_csv('data/Telco-Customer-Churn.csv')

preprocessing_pipeline = build_preprocessing_pipeline()

TARGET = 'Churn'

X = df.drop(columns=[TARGET, 'customerID'])
y = df[TARGET].map({'Yes': 1, 'No': 0})

X_processed = preprocessing_pipeline.fit_transform(X)
logging.info(f"Preprocessing complete. Feature shape: {X_processed.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y.values, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = ChurnMLP(input_dim=X_train_tensor.shape[1]).to(device)

# Compute positive class weight
pos_weight = torch.tensor([len(y_train) / y_train.sum() - 1]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# Training loop
# ------------------------
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)

    epoch_loss /= len(train_loader.dataset)

    if (epoch + 1) % 10 == 0:
        logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

# ------------------------
# Evaluation
# ------------------------
model.eval()
with torch.no_grad():
    test_logits = model(X_test_tensor)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs > 0.5).float()

y_true = y_test_tensor.cpu().numpy()
y_pred = test_preds.cpu().numpy()

logging.info(f"Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
logging.info("Classification Report:\n" + classification_report(y_true, y_pred))
logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred)))

MODEL_PATH = "models/churnMLP.pth"

torch.save({
    "model_state": model.state_dict(),
    "input_dim": X_train_tensor.shape[1]
}, "models/churnMLP.pth")

joblib.dump(preprocessing_pipeline, "models/preprocessing.joblib")

logging.info(f"Model and preprocessing pipeline saved to {MODEL_PATH} and models/preprocessing.joblib")
