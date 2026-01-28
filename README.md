This project focuses around Teclo-Customer-Churn data.
I trained a model after preprocessing said data, and created my own database where I add random customers.
I use the trained model to then try to predict if these customers would churn or not.


MODEL:

Classification Report:
              precision    recall  f1-score   support

         0.0       0.91      0.77      0.83      1036
         1.0       0.55      0.79      0.65       373

    accuracy                           0.77      1409
   macro avg       0.73      0.78      0.74      1409
weighted avg       0.81      0.77      0.78      1409

Confusion Matrix:
[[798 238]
 [ 80 293]]

 Next steps for this project would be to try and improve these results, to hopefully improve the model accuracy.

 HOW TO USE

 python -m venv venv
 venv\Scripts\activate

 pip install -r requirements.txt

 python database/init_db.py

 python main.py

