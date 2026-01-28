import pandas as pd

def save_customers(database, df: pd.DataFrame):
    """
    Save generated customers into the database.
    If a customerID already exists, ignore it.
    """
    with database as db:
        for _, row in df.iterrows():
            try:
                db.cursor.execute("""
                    INSERT OR IGNORE INTO customers (
                        customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
                        PhoneService, MultipleLines, InternetService, OnlineSecurity,
                        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                        MonthlyCharges, TotalCharges
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(row.values))
            except Exception as e:
                print(f"Failed to insert customer {row['customerID']}: {e}")

        db.commit()
        print(f"{len(df)} customers processed and saved to the database.")