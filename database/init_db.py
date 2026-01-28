from db import CustomerDB

def initialize_db():

    with CustomerDB() as db:
        customer_query = """
        CREATE TABLE IF NOT EXISTS customers (
            customerID TEXT PRIMARY KEY,
            gender TEXT,
            SeniorCitizen INTEGER,
            Partner TEXT,
            Dependents TEXT,
            tenure INTEGER,
            PhoneService TEXT,
            MultipleLines TEXT,
            InternetService TEXT,
            OnlineSecurity TEXT,
            OnlineBackup TEXT,
            DeviceProtection TEXT,
            TechSupport TEXT,
            StreamingTV TEXT,
            StreamingMovies TEXT,
            Contract TEXT,
            PaperlessBilling TEXT,
            PaymentMethod TEXT,
            MonthlyCharges REAL,
            TotalCharges REAL
        );
    """

        prediction_query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customerID TEXT,
            ChurnPrediction TEXT,
            ChurnProbability REAL,
            PredictionDate TEXT,
            FOREIGN KEY (customerID) REFERENCES customers(customerID)
        );
    """
        db.cursor.execute("PRAGMA foreign_keys = ON;")
        db.cursor.execute(customer_query)
        db.cursor.execute(prediction_query)
        db.commit()

initialize_db()
