import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import shap
import streamlit.components.v1 as components
import xgboost as xgb 


st.title('Dashboard client validation')
X_train_dashboard = pd.read_csv('./X_train_dashboard.csv').drop(columns=['SK_ID_CURR'])
y_train_dashboard = pd.read_csv('./y_train_dashboard.csv').drop(columns=['SK_ID_CURR'])
X_test_dashboard = pd.read_csv('./X_test_dashboard.csv')
all_client_numbers = X_test_dashboard['SK_ID_CURR']
number_client = st.selectbox('Choose a client number :', all_client_numbers)
model_xgb = xgb.XGBClassifier(
    learning_rate = 0.13340120276728681,
    max_depth = 43,
    min_child_weight = 1.5589983674569985,
    reg_alpha = 0.08048825852598339,
    reg_lambda = 0.006187471945504702,
    seed = 42
)
model_xgb.fit(X_train_dashboard, y_train_dashboard)
X_test = X_test_dashboard.loc[X_test_dashboard['SK_ID_CURR']==number_client].drop(columns=['SK_ID_CURR'])
limit_proba = 0.8


# To obtain client score
X_test = X_test_dashboard.loc[X_test_dashboard['SK_ID_CURR']==number_client].drop(columns=['SK_ID_CURR'])
proba = model_xgb.predict_proba(X_test)[0][0]
client_score = proba


# To print the indicator plot of the score
st.header('Indicator for the validation of client score :')
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = client_score,
    mode = "gauge+number+delta",
    title = {'text': "Indicator client score"},
    gauge = {'axis': {'range': [None, 1]},
             'bar': {'color': "#FF0000"},
             'steps' : [
                 {'range': [0, limit_proba], 'color': "#FFFFFF"},
                 {'range': [limit_proba, 1], 'color': "#98FB98"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': limit_proba}}))
st.plotly_chart(fig)


# To print the client status (Validation of the client)
if client_score > limit_proba:
    client_status_result = 'Good client, credit is accepted'    
else:
    client_status_result = 'Bad client, credit is not accepted'    
st.write(client_status_result)
st.markdown('The white area is where the score less than 0.8, and the green area is the validation area where the score is more than 0.8.')
st.markdown('To accept credit for a client, the client must has a score more than 0.8. That means the indicator reaches the green area.')


# To print feature importance local with shap
st.header('Local feature importance with shap :')
shap_explainer = shap.TreeExplainer(model_xgb)
shap_values = shap_explainer.shap_values(X_train_dashboard, check_additivity=False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(shap.plots.force(shap_explainer.expected_value, shap_values[1], X_test, matplotlib=True))

