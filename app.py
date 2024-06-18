import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# read in data
df = pd.read_csv('V:/processed_data.csv')


# Check for and drop any missing values
df = df.dropna()

# Split the data into features and target
x = df.drop('stroke', axis=1)
y = df['stroke']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Build an SVM Classifier with probability estimates
model = SVC(probability=True, random_state=42)

# Train the model
model.fit(x_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(x_test_scaled)
y_prob = model.predict_proba(x_test_scaled)[:, 1]  # Probability of the positive class (stroke)

# Create the Streamlit app
with st.form("stroke_form"):
    st.markdown('<p style="color:#8b0000; font-size: 36px;"><b>Stroke Prediction Quiz</b></p>', unsafe_allow_html=True)
    '##### Answer the questions below to determine your potential risk of experiencing a stroke.'

    age = st.number_input('What is your age?', min_value=0)
    average_glucose = st.number_input('What is your average glucose level?', min_value=0.0)
    bmi = st.number_input('What is your BMI?', min_value=0.0)
    gender = st.selectbox('What is your gender?', ['Female', 'Male'])
    ever_married = st.selectbox('Have you ever been married?', ['No', 'Yes'])
    work_type = st.selectbox('What is your work type?', ['Never worked', 'Private', 'Self-employed', 'Children', 'Government'])
    residence_type = st.selectbox('Do you live in an urban or rural area?', ['Rural', 'Urban'])
    smoking_status = st.selectbox('Do you smoke?', ['Never smoked', 'Formerly smoked', 'Smokes'])
    age_group = st.selectbox('What is your age group?', ['Adult', 'Senior', 'Teen', 'Toddler'])

    submitted = st.form_submit_button("Submit")
    if submitted:
        user_data = {
            'age': age,
            'avg_glucose_level': average_glucose,
            'bmi': bmi,
            'gender_Male': 1 if gender == 'Male' else 0,
            'ever_married_Yes': 1 if ever_married == 'Yes' else 0,
            'work_type_Never_worked': 1 if work_type == 'Never worked' else 0,
            'work_type_Private': 1 if work_type == 'Private' else 0,
            'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
            'work_type_children': 1 if work_type == 'Children' else 0,
            'work_type_Government': 1 if work_type == 'Government' else 0,
            'Residence_type_Urban': 1 if residence_type == 'Urban' else 0,
            'smoking_status_formerly smoked': 1 if smoking_status == 'Formerly smoked' else 0,
            'smoking_status_never smoked': 1 if smoking_status == 'Never smoked' else 0,
            'smoking_status_smokes': 1 if smoking_status == 'Smokes' else 0,
            'age_group_Senior': 1 if age_group == 'Senior' else 0,
            'age_group_Teen': 1 if age_group == 'Teen' else 0,
            'age_group_Toddler': 1 if age_group == 'Toddler' else 0,
            'age_group_Adult': 1 if age_group == 'Adult' else 0
        }

        user_data_df = pd.DataFrame([user_data], columns=x.columns)

        # Scale the user data
        user_data_scaled = scaler.transform(user_data_df)

        # Predict probability of stroke
        prediction_prob = model.predict_proba(user_data_scaled)[0, 1]  # Probability of having a stroke

        '### Prediction:'
        st.write(f'You have a {prediction_prob * 100:.2f}% probability of having a stroke.')