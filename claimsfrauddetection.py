# Using 'streamlit' library to create a health claims fraud detection web app
# Import necessary libraries
import numpy as np # for data manipulation
import pickle # for loading the saved model
import streamlit as st # for deployment
import pandas as pd # for numerical computation
import matplotlib.pyplot as plt # for data visualization
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost
from tqdm import tqdm

# Set title of the web app
st.title('Health Insurance Claims Fraud Detection')

# Loading the cleaned claim data
# data = pd.read_csv('C:/claims_fraud_datasets/cleanedclaim_data.csv')

# creating a sidebar for adjusting the test size
st.subheader('XGBoost Classifier')

# Load the trained XGBoost model
xgb_model = pickle.load(open('C:/Isaac/trained_xgb_model.sav', 'rb'))

# Data cleanup and feature engineering
def preprocess_data(input_data):
    # Drop unnecessary columns
    ID_Columns = ['BeneID', 'ClaimID', 'Provider']
    input_data.drop(columns = ID_Columns, inplace = True, errors = 'ignore')

    # Replace values in RenalDiseaseIndicator
    input_data['RenalDiseaseIndicator'].replace(to_replace = ['Y', '0'], value = [1, 0], inplace = True)

    # Replace values in Chronic condition features
    for ChronicCondCol in [col for col in list(input_data.columns) if 'ChronicCond' in col]:
        input_data[ChronicCondCol].replace(to_replace = 2, value = 0, inplace = True)

    # Replace class labels of the Gender feature
    input_data['Gender'].replace(to_replace = 2, value = 0, inplace=True)

    # Drop columns with high percentages of missing values
    cols = ['ClmProcedureCode_2', 'ClmProcedureCode_3',
            'ClmProcedureCode_4', 'ClmProcedureCode_5',
            'ClmProcedureCode_6', 'ClmDiagnosisCode_10']
    input_data.drop(columns = cols, inplace = True, errors = 'ignore')

    # Fill missing values in 'DeductibleAmtPaid'
    input_data['DeductibleAmtPaid'].fillna(0, inplace = True)

    # Feature engineering
    ReimDedAmtCols = [col for col in input_data.columns if 'Amt' in col]
    input_data['TotalClaimAmt'] = input_data['InscClaimAmtReimbursed'] + input_data['DeductibleAmtPaid']
    input_data['IPTotalAmt'] = input_data['IPAnnualReimbursementAmt'] + input_data['IPAnnualDeductibleAmt']
    input_data['OPTotalAmt'] = input_data['OPAnnualReimbursementAmt'] + input_data['OPAnnualDeductibleAmt']
    input_data.drop(columns = ReimDedAmtCols, inplace = True)

    DateColumns = [col for col in input_data.columns if ('Dt' in col or 'DOB' in col or 'DOD' in col)]
    input_data[DateColumns] = input_data[DateColumns].apply(pd.to_datetime)
    input_data['IsAlive'] = input_data['DOD'].apply(lambda val: 0 if pd.notnull(val) else 1)

    DateMax = max(input_data['ClaimEndDt'].max(), input_data['DischargeDt'].max())
    input_data['Age'] = input_data.apply(
        lambda val: round(((val['DOD'] - val['DOB']).days)/365) if pd.notnull(val['DOD'])
        else round(((DateMax - val['DOB']).days)/365), axis = 1)
    
    input_data['ClaimSettlementDuration'] = (input_data['ClaimEndDt'] - input_data['ClaimStartDt']).dt.days
    input_data['AdmissionDuration'] = (input_data['DischargeDt'] - input_data['AdmissionDt']).dt.days.fillna(0).astype(int)

    # Adding inpatient and outpatient indicator feature; Admitted
    input_data['Admitted'] = np.where(input_data['AdmissionDt'].notnull(), 1, 0)
    input_data.drop(columns = DateColumns, inplace = True, errors = 'ignore')

    PhysicianColumns = [col for col in input_data.columns if 'Physician' in col]
    input_data['NumUniquePhysician'] = input_data[PhysicianColumns]\
        .apply(lambda val: len(set([Physician for Physician in val if pd.notnull(Physician)])), axis=1)
    input_data['NumPhysicianRole'] = input_data[PhysicianColumns]\
        .apply(lambda val: len([Physician for Physician in val if pd.notnull(Physician)]), axis = 1)
    input_data['SamePhysicianMultipleRole1'] = input_data[['NumUniquePhysician', 'NumPhysicianRole']]\
        .apply(lambda val: 1 if val['NumUniquePhysician'] == 1 and val['NumPhysicianRole'] > 1 else 0, axis = 1)
    input_data['SamePhysicianMultipleRole2'] = input_data[['NumUniquePhysician', 'NumPhysicianRole']]\
        .apply(lambda val: 1 if val['NumUniquePhysician'] == 2 and val['NumPhysicianRole'] > 2 else 0, axis = 1)

    # Dropping the original features related to Physicians
    input_data.drop(columns = PhysicianColumns, inplace = True, errors = 'ignore')

    # Replacing Claim Diagnosis Codes with 1 if the value is not null, else 0
    ClaimDiagCodeColumns = [col for col in input_data.columns if 'ClmDiagnosisCode' in col]
    for code in tqdm(ClaimDiagCodeColumns):
        input_data[code] = input_data[code].apply(lambda val: 1 if not pd.isnull(val) else 0)

    # Encode ClaimProcedureCode_1
    for code in tqdm(['ClmProcedureCode_1']):
        input_data[code] = input_data[code].apply(lambda val: 1 if not pd.isnull(val) else 0)

    # Encode ClmAdmitDiagnosisCode
    for code in tqdm(['ClmAdmitDiagnosisCode']):
        input_data[code] = input_data[code].apply(lambda val: 1 if not pd.isnull(val) else 0)

    # Encode DiagnosisGroupCode
    for code in tqdm(['DiagnosisGroupCode']):
        input_data[code] = input_data[code].apply(lambda val: 1 if not pd.isnull(val) else 0)

    # Drop highly correlated features
    cols_to_drop = ['DiagnosisGroupCode', 'AdmissionDuration', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_7',
                    'ClmDiagnosisCode_6', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_4', 'ClmProcedureCode_1']
    input_data.drop(columns = cols_to_drop, inplace = True, errors = 'ignore')

    # Print columns after preprocessing
    st.write('Columns after preprocessing:', input_data.columns)

    return input_data

# Function for fraud detection
def fraud_detection(user_input):
    # Preprocess user input
    user_input = user_input.copy()
    user_input = preprocess_data(pd.DataFrame([user_input]))

    # Convert input data to numpy array
    user_input = np.asarray(user_input)
    # Reshape the array - prediction is done for one instance
    user_input = user_input.reshape(1, -1)

    # Make predictions
    prediction = (xgb_model.predict_proba(user_input)[:, 1] >= 0.25).astype(int)

    if prediction[0] == 1:
        return 'Potential Fraud Detected!'
    else:
        return 'No Fraud Detected.'
    
# Getting the input data from the user for fraud prediction
def main():
    st.sidebar.subheader('Claim Information') # adding subheader
    # Getting the input data from the user
    BeneID = st.sidebar.text_input('Beneficiary ID')
    ClaimID = st.sidebar.text_input('Claim ID')
    ClaimStartDt = st.sidebar.text_input('Claim start date')
    ClaimEndDt = st.sidebar.text_input('Claim end date')
    Provider = st.sidebar.text_input('Provider ID')
    AttendingPhysician = st.sidebar.text_input('Attending physician ID')
    OperatingPhysician = st.sidebar.text_input('Operating physician ID')
    OtherPhysician = st.sidebar.text_input('Other physician ID')
    AdmissionDt = st.sidebar.text_input('Patient\'s admision date')
    DischargeDt = st.sidebar.text_input('Patient\'s discharge date')
    InscClaimAmtReimbursed = st.sidebar.text_input('Insurance claim amount reimbursed')
    DeductibleAmtPaid = st.sidebar.text_input('Deductible amount paid')
    IPAnnualReimbursementAmt = st.sidebar.text_input('Inpatient annual reimbursement amount')
    IPAnnualDeductibleAmt = st.sidebar.text_input('Inpatient annual deductible amount')
    OPAnnualReimbursementAmt = st.sidebar.text_input('Outpatient annual reimbursement amount')
    OPAnnualDeductibleAmt = st.sidebar.text_input('Outpatient annual deductible amount')
    ClmAdmitDiagnosisCode = st.sidebar.text_input('Claim admission diagnosis code')
    DiagnosisGroupCode = st.sidebar.text_input('Diagnosis group code')
    ClmDiagnosisCode_1 = st.sidebar.text_input('Claim diagnosis code_1')
    ClmDiagnosisCode_2 = st.sidebar.text_input('Claim diagnosis code_2')
    ClmDiagnosisCode_3 = st.sidebar.text_input('Claim diagnosis code_3')
    ClmDiagnosisCode_4 = st.sidebar.text_input('Claim diagnosis code_4')
    ClmDiagnosisCode_5 = st.sidebar.text_input('Claim diagnosis code_5')
    ClmDiagnosisCode_6 = st.sidebar.text_input('Claim diagnosis code_6')
    ClmDiagnosisCode_7 = st.sidebar.text_input('Claim diagnosis code_7')
    ClmDiagnosisCode_8 = st.sidebar.text_input('Claim diagnosis code_8')
    ClmDiagnosisCode_9 = st.sidebar.text_input('Claim diagnosis code_9')
    ClmDiagnosisCode_10 = st.sidebar.text_input('Claim diagnosis code_10')
    ClmProcedureCode_1 = st.sidebar.text_input('Claim procedure code_1')
    ClmProcedureCode_2 = st.sidebar.text_input('Claim procedure code_2')
    ClmProcedureCode_3 = st.sidebar.text_input('Claim procedure code_3')
    ClmProcedureCode_4 = st.sidebar.text_input('Claim procedure code_4')
    ClmProcedureCode_5 = st.sidebar.text_input('Claim procedure code_5')
    ClmProcedureCode_6 = st.sidebar.text_input('Claim procedure code_6')
    DOB = st.sidebar.text_input('Patient\'s date of birth')
    DOD = st.sidebar.text_input('Patient\'s date of death')
    Gender = st.sidebar.text_input('Patient\'s gender')
    RenalDiseaseIndicator = st.sidebar.text_input('Renal disease indicator')
    ChronicCond_Alzheimer = st.sidebar.text_input('Chronic condition - Alzheimer disease value')
    ChronicCond_Heartfailure = st.sidebar.text_input('Chronic condition - Heartfailure disease value')
    ChronicCond_KidneyDisease = st.sidebar.text_input('Chronic condition - Kidney disease Value')
    ChronicCond_Cancer = st.sidebar.text_input('Chronic condition - Cancer disease value')
    ChronicCond_ObstrPulmonary = st.sidebar.text_input('Chronic condition - Obstructive pulmonary disease value')
    ChronicCond_Depression = st.sidebar.text_input('Chronic condition - Depression disease value')
    ChronicCond_Diabetes = st.sidebar.text_input('Chronic condition - Diabetes disease value')
    ChronicCond_IschemicHeart = st.sidebar.text_input('Chronic condition - Ischemic heart disease value')
    ChronicCond_Osteoporasis = st.sidebar.text_input('Chronic condition - Osteoporasis disease value')
    ChronicCond_rheumatoidarthritis = st.sidebar.text_input('Chronic condition - Rheumatoid arthritis disease value')
    ChronicCond_stroke = st.sidebar.text_input('Chronic condition - Stroke disease value')
    PhysCode_PHY330576 = st.sidebar.text_input('Physician code_PHY330576')
    PhysCode_PHY412132 = st.sidebar.text_input('Physician code_PHY412132')
    PhysCode_PHY341578 = st.sidebar.text_input('Physician code_PHY341578')
    PhysCode_PHY337425 = st.sidebar.text_input('Physician code_PHY337425')
    ClaimDiagCode_4019 = st.sidebar.text_input('Claim diagnosis code_4019')
    ClaimDiagCode_25000 = st.sidebar.text_input('Claim diagnosis code_25000')
    ClaimDiagCode_2724 = st.sidebar.text_input('Claim diagnosis code_2724')
    ClaimDiagCode_V5869 = st.sidebar.text_input('Claim diagnosis code_V5869')
    ClaimDiagCode_4011 = st.sidebar.text_input('Claim diagnosis code_4011')
    ClaimDiagCode_42731 = st.sidebar.text_input('Claim diagnosis code_42731')
    ClaimDiagCode_V5861 = st.sidebar.text_input('Claim diagnosis code_V5861')
    ClaimDiagCode_2720 = st.sidebar.text_input('Claim diagnosis code_2720')
    ClaimDiagCode_2449 = st.sidebar.text_input('Claim diagnosis code_2449')
    ClaimDiagCode_4280 = st.sidebar.text_input('Claim diagnosis code_4280')

    user_input = {'BeneID':BeneID, 'ClaimID':ClaimID, 'ClaimStartDt':ClaimStartDt, 'ClaimEndDt':ClaimEndDt, 'Provider':Provider,
                  'AttendingPhysician':AttendingPhysician, 'OperatingPhysician':OperatingPhysician, 'OtherPhysician':OtherPhysician,
                  'AdmissionDt':AdmissionDt, 'DischargeDt':DischargeDt, 'InscClaimAmtReimbursed':InscClaimAmtReimbursed,
                  'DeductibleAmtPaid':DeductibleAmtPaid, 'IPAnnualReimbursementAmt':IPAnnualReimbursementAmt, 'DOB':DOB, 'DOD':DOD,
                  'IPAnnualDeductibleAmt':IPAnnualDeductibleAmt, 'OPAnnualReimbursementAmt':OPAnnualReimbursementAmt,
                  'OPAnnualDeductibleAmt':OPAnnualDeductibleAmt, 'ClmAdmitDiagnosisCode':ClmAdmitDiagnosisCode, 
                  'DiagnosisGroupCode':DiagnosisGroupCode, 'ClmDiagnosisCode_1':ClmDiagnosisCode_1, 'ClmDiagnosisCode_2':ClmDiagnosisCode_2,
                  'ClmDiagnosisCode_3':ClmDiagnosisCode_3, 'ClmDiagnosisCode_4':ClmDiagnosisCode_4, 'ClmDiagnosisCode_5':ClmDiagnosisCode_5,
                  'ClmDiagnosisCode_6':ClmDiagnosisCode_6, 'ClmDiagnosisCode_7':ClmDiagnosisCode_7, 'ClmDiagnosisCode_8':ClmDiagnosisCode_8,
                  'ClmDiagnosisCode_9':ClmDiagnosisCode_9, 'ClmDiagnosisCode_10':ClmDiagnosisCode_10, 'ClmProcedureCode_1':ClmProcedureCode_1,
                  'ClmProcedureCode_2':ClmProcedureCode_2, 'ClmProcedureCode_3':ClmProcedureCode_3, 'ClmProcedureCode_4':ClmProcedureCode_4,
                  'ClmProcedureCode_5':ClmProcedureCode_5, 'ClmProcedureCode_6':ClmProcedureCode_6, 'Gender':Gender, 
                  'RenalDiseaseIndicator':RenalDiseaseIndicator, 'ChronicCond_Alzheimer':ChronicCond_Alzheimer, 
                  'ChronicCond_Heartfailure':ChronicCond_Heartfailure, 'ChronicCond_KidneyDisease':ChronicCond_KidneyDisease,
                  'ChronicCond_Cancer':ChronicCond_Cancer, 'ChronicCond_ObstrPulmonary':ChronicCond_ObstrPulmonary,
                  'ChronicCond_Depression':ChronicCond_Depression, 'ChronicCond_Diabetes':ChronicCond_Diabetes, 
                  'ChronicCond_IschemicHeart':ChronicCond_IschemicHeart, 'ChronicCond_Osteoporasis':ChronicCond_Osteoporasis,
                  'ChronicCond_rheumatoidarthritis':ChronicCond_rheumatoidarthritis, 'ChronicCond_stroke':ChronicCond_stroke,
                  'PhysCode_PHY330576':PhysCode_PHY330576, 'PhysCode_PHY412132':PhysCode_PHY412132, 'PhysCode_PHY341578':PhysCode_PHY341578,
                  'PhysCode_PHY337425':PhysCode_PHY337425, 'ClaimDiagCode_4019':ClaimDiagCode_4019, 'ClaimDiagCode_25000':ClaimDiagCode_25000,
                  'ClaimDiagCode_2724':ClaimDiagCode_2724, 'ClaimDiagCode_V5869':ClaimDiagCode_V5869, 'ClaimDiagCode_4011':ClaimDiagCode_4011,
                  'ClaimDiagCode_42731':ClaimDiagCode_42731, 'ClaimDiagCode_V5861':ClaimDiagCode_V5861, 'ClaimDiagCode_2720':ClaimDiagCode_2720,
                  'ClaimDiagCode_2449':ClaimDiagCode_2449, 'ClaimDiagCode_4280':ClaimDiagCode_4280
                 }

    # Add a 'Detect Fraud' button
    if st.button('Detect Fraud'):
        # Making prediction
        prediction = fraud_detection(user_input)

        # Showing the results
        st.success(f'Fraud Detection Result: {prediction}')

if __name__ == '__main__':
    main()