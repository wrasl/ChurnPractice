import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database.db import CustomerDB
from utils.save_customers import save_customers
from utils.generate_customers import generate_customers
from src.predict import predict_churn
import pandas as pd
from datetime import datetime

if __name__ == "__main__":

    df = generate_customers(n=30)
    save_customers(CustomerDB(), df)

    with CustomerDB() as db:
        df_to_predict = pd.read_sql("""
            SELECT *
            FROM customers
            WHERE customerID NOT IN (SELECT customerID FROM predictions)
        """, db.conn)

    results = predict_churn("models/churnMLP.pth", df_to_predict)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with CustomerDB() as db:
        records = [
            (cid, churn, float(prob), timestamp)
            for cid, churn, prob in zip(results['customerID'], results['Churn'], results['Churn_Probability'])
        ]
        db.cursor.executemany("""
            INSERT INTO predictions (customerID, ChurnPrediction, ChurnProbability, PredictionDate)
            VALUES (?, ?, ?, ?)
        """, records)
        db.commit()

    csv_filename = f"data/predictions/predictions-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    results.to_csv(csv_filename, index=False)
