import streamlit as st
import pandas as pd
from flask import Flask, request
import requests
import plotly.graph_objects as go
import shap
import joblib
import streamlit.components.v1 as components



st.title('Dashboard client validation')
X_train_dashboard = pd.read_csv('./X_train_dashboard.csv').drop(columns=['SK_ID_CURR'])
X_test_dashboard = pd.read_csv('./X_test_dashboard.csv')
all_client_numbers = X_test_dashboard['SK_ID_CURR']
number_client = st.selectbox('Choose a client number :', all_client_numbers)
filename = "model_xgb_dashb.joblib"
loaded_model = joblib.load(filename)
X_test = X_test_dashboard.loc[X_test_dashboard['SK_ID_CURR']==number_client].drop(columns=['SK_ID_CURR'])
limit_proba = 0.8


# To obtain client score
def client_score_response():
    '''_Use the function client_score_response() to get the client score from backend_.
    Args:
        none.  
                    
    Returns:
        client_score_result: _Client score by his number_.
    '''
    response = requests.post('http://localhost:6000/client_score', json={'number_client': number_client})
    client_score_result = response.json()
    return client_score_result
client_score = client_score_response()


# To print the indicator plot of the score
st.header('Indicator for the validation of client score :')
limit_proba = 0.8
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
def client_status_response():
    '''_Use the function client_status_response() to get the client status from backend_.
    Args:
        none.  
                    
    Returns:
        client_status_result: _Client status by his score._.
    '''
    response = requests.post('http://localhost:6000/client_status', json={'client_score': client_score})
    client_status_result = response.json()
    return client_status_result
client_status_result = client_status_response()
st.write(client_status_result)
st.markdown('The white area is where the score less than 0.8, and the green area is the validation area where the score is more than 0.8.')
st.markdown('To accept credit for a client, the client must has a score more than 0.8. That means the indicator reaches the green area.')


# To print feature importance local with shap
st.header('Local feature importance with shap :')
# The function st_shap() helps to print the feature importance with shap (copied from internet)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
shap_explainer = shap.TreeExplainer(loaded_model)
shap_values = shap_explainer.shap_values(X_train_dashboard, check_additivity=False)
st_shap(shap.force_plot(shap_explainer.expected_value, shap_values[1], X_test))
st.markdown('You can move the cursor to see features names and theirs scores')

