import numpy as np
import pandas as pd
import pickle

# gender = 1
# married = 1
# dependents = 0
# education = 0
# self_employed = 0
# applicant_income = 40000
# coapplicant_income = 0
# loan_amount = 15000
# loan_term = 360
# credit_history_available = 1
# housing = 1
# locality = 1

input_data = {
    'Gender': [1],
    'Married': [0],
    'Dependents': [0],
    'Education': [1],
    'Self_Employed': [0],
    'ApplicantIncome': [100000],
    'CoapplicantIncome': [0],
    'LoanAmount': [150],
    'Loan_Term': [360],
    'Credit_History_Available': [1],
    'Housing': [1],
    'Locality': [1]
}

input_df = pd.DataFrame(input_data)

loaded_model = pd.read_pickle(open('randomforest_model.pkl', 'rb'))

predicted_risk = loaded_model.predict(input_df)
print(predicted_risk)