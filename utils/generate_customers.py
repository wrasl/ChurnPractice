import pandas as pd
import numpy as np
import string
import datetime

def random_customer_ids(n):

    ids = []
    for _ in range(n):
        digits = np.random.randint(0, 10000)
        letters = ''.join(np.random.choice(list(string.ascii_uppercase), size=4))

        ids.append(f"{digits:04d}-{letters}")
    return ids


def generate_customers(n=25):

    customerIDs = random_customer_ids(n)
    
    # Binary / Yes-No columns
    yes_no = lambda size: np.random.choice(['Yes', 'No'], size=size)
    gender = np.random.choice(['Male', 'Female'], size=n)
    senior = np.random.choice([0, 1], size=n)
    
    # Numeric columns
    tenure = np.random.randint(0, 72, size=n)           # 0-72 months
    monthly_charges = np.round(np.random.uniform(20, 120, size=n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(-10, 10, size=n), 2)
    
    # Categorical multi-class columns
    multiple_lines = np.random.choice(['No', 'Yes', 'No phone service'], size=n)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n)
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], size=n)
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'], size=n)
    device_protection = np.random.choice(['Yes', 'No', 'No internet service'], size=n)
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], size=n)
    streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], size=n)
    streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], size=n)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n)
    paperless_billing = yes_no(n)
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], size=n)
    phone_service = yes_no(n)
    partner = yes_no(n)
    dependents = yes_no(n)

    df = pd.DataFrame({
        'customerID': customerIDs,
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    })

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = f"Random-Customers-{timestamp}.csv"
    df.to_csv(f"data/customers/{filename}", index=False)

    return df
