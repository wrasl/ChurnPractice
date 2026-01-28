import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import pandas as pd
import numpy as np
from model import ChurnMLP
import logging
import joblib

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)

# -------------------- Functions --------------------
def predict_churn(model_checkpoint_path, df_customers, device=None):
    """
    Run churn prediction on a dataframe of customers.

    Args:
        model_checkpoint_path (str): Path to saved PyTorch model checkpoint.
        df_customers (pd.DataFrame): Customers to predict (must have all features expected by model).
        device (torch.device, optional): Torch device to use (cpu or cuda). Defaults to auto-detect.

    Returns:
        pd.DataFrame: customerID, Churn (Yes/No), Churn_Probability (rounded 3 decimals)
    """
    if df_customers.empty:
        logging.info("No customers provided for prediction.")
        return pd.DataFrame(columns=["customerID", "Churn", "Churn_Probability"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoints and model
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    preprocessing_pipeline = joblib.load("models/preprocessing.joblib")

    model = ChurnMLP(input_dim=checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logging.info("Model and preprocessing pipeline loaded. Model set to evaluation mode.")

    # Preprocess
    X = preprocessing_pipeline.transform(df_customers)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    logging.info(f"Preprocessed {len(df_customers)} customers for prediction.")

    # Predict
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        probs_rounded = np.round(probs, 3)

    results = pd.DataFrame({
        "customerID": df_customers["customerID"],
        "Churn": ["Yes" if p == 1 else "No" for p in preds],
        "Churn_Probability": probs_rounded
    })

    logging.info("Predictions completed.")
    return results
