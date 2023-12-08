import streamlit as st
import pandas as pd
import json
import datetime
import time
import plotly.graph_objects as go
import requests


#####################################################
# STYLING AND PAGE CONFIGURATION
#####################################################

st.set_page_config(page_title='Electricity Prediction', layout='wide', page_icon=':zap:')

# Define the style for the spinning effect
st.markdown("""
    <style>
    @keyframes spin { 
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .spinning {
        display: inline-block;
        animation: spin 10s linear infinite;
    }
    </style>
    """, unsafe_allow_html=True)


# CSS to center and position the title higher on the page
st.markdown("""
    <style>
    body {
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    h1 {
        text-align: center;
        margin-top: -50px;  /* Adjust the negative value to move title higher */
    }
    </style>
    """, unsafe_allow_html=True)

# Centered title using markdown with HTML
st.markdown('<h1><span class="spinning">⚡️</span> Electricity Consumption and Production Prediction <span class="spinning">⚡️</span></h1>', unsafe_allow_html=True)

#####################################################
# DATA LOADING FUNCTIONS
#####################################################

def load_pred_id_mapping():
    """Load the prediction unit ID mapping from a JSON file."""
    with open('streamlit/data/mapping_prediction_unit_id.json', 'r') as file:
        pred_id_mapping = json.load(file)
    return pd.DataFrame.from_dict(pred_id_mapping, orient='index')

# Load the y_true data
y_true = pd.read_csv('streamlit/data/y_true.csv')
y_true['datetime'] = pd.to_datetime(y_true['datetime'])
pred_id_mapping = load_pred_id_mapping()

# Define county and product type mappings
county_name_mapping = {
    "0": "Harjumaa", "1": "Hiiumaa", "2": "Ida-Virumaa", "3": "Järvamaa",
    "4": "Jõgevamaa", "5": "Lääne-Virumaa", "6": "Läänemaa", "7": "Pärnumaa",
    "8": "Põlvamaa", "9": "Raplamaa", "10": "Saaremaa", "11": "Tartumaa",
    "13": "Valgamaa", "14": "Viljandimaa", "15": "Võrumaa"
}
product_type_mapping = {'0': "Combined", '1': "Fixed", '2': "General service", '3': "Spot"}

#####################################################
# UTILITY FUNCTIONS
#####################################################

def get_county_number_from_name(county_name):
    """Convert county name to its corresponding number."""
    for num, name in county_name_mapping.items():
        if name == county_name:
            return num
    return None

def get_product_type_number(selected_product_name):
    """Convert product type name to its corresponding number."""
    for num, name in product_type_mapping.items():
        if name == selected_product_name:
            return num
    return None

def find_pred_id(is_business, county_number, selected_product_type):
    """Find prediction unit ID based on business flag, county number, and product type."""
    is_business_flag = 1 if is_business == 'Yes' else 0
    product_type_number = get_product_type_number(selected_product_type)
    for _, row in pred_id_mapping.iterrows():
        if row['county'] == int(county_number) and \
           row['is_business'] == is_business_flag and \
           row['product_type'] == int(product_type_number):
            return row['prediction_unit_id']
    return None

def get_available_product_types(county_number, is_business):
    """Get available product types for a given county number and business status."""
    is_business_flag = 1 if is_business == 'Yes' else 0
    filtered_df = pred_id_mapping[(pred_id_mapping['county'] == int(county_number)) & 
                                  (pred_id_mapping['is_business'] == is_business_flag)]
    product_type_numbers = filtered_df['product_type'].unique().tolist()
    return [product_type_mapping[str(num)] for num in product_type_numbers]

def predict_electricity(start_date, end_date, prediction_unit_id, model):
    """Function to predict electricity consumption and production."""
    # Fetch forecast data from the API
    forecast_conso, forecast_prod = fetch_forecast_from_api(prediction_unit_id, model)
    return forecast_conso, forecast_prod


####################################################
# API CALL
#################################################### 

def fetch_forecast_from_api(prediction_unit_id, model):
    """Fetch forecast data from the API based on the prediction unit ID and model."""
    api_url = f"https://enefitservice-gipozgf35q-ew.a.run.app/predict?PUID={prediction_unit_id}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        # st.write("API Response:", data)
        try:
            if model == 'Prophet':
                return pd.DataFrame(data['prophet_conso'][str(prediction_unit_id)]), \
                       pd.DataFrame(data['prophet_prod'][str(prediction_unit_id)])
            elif model == 'Auto-SARIMA':
                conso_df = pd.DataFrame({'ds': data['forecast_conso']['ds'], 'yhat': data['forecast_conso']['MSTL']})
                prod_df = pd.DataFrame({'ds': data['forecast_prod']['ds'], 'yhat': data['forecast_prod']['MSTL']})
                return conso_df, prod_df
        except KeyError as e:
            st.error(f"KeyError occurred: {e}")
            return None, None
    else:
        st.error("Failed to fetch data from API")
        return None, None


#####################################################
# PLOTTING FUNCTIONS
#####################################################

def create_plotly_chart(y_true, y_pred, title, y_label, showlegend=True):
    """Create a Plotly chart for visualizing actual and predicted values."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true['datetime'], 
        y=y_true['target'],
        mode='lines',
        name='Actual',
        line=dict(color='darkblue')  
    ))

    # Plot predicted values
    if y_pred is not None:
        fig.add_trace(go.Scatter(
            x=y_pred['ds'],  
            y=y_pred['yhat'], 
            mode='lines',
            name='Predicted',
            line=dict(color='red')  
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        plot_bgcolor="white",  
        paper_bgcolor="white",  
        font=dict(color="black"),
        title_font=dict(size=20, color="black"),
        title_x=0.35,
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            gridcolor= 'lightgrey',
            linecolor="black",
            linewidth=2,
            ticks='outside',
            tickfont=dict(family='Arial', size=12, color='black'),
            title_font=dict(color="black"),
            tickmode='auto',
            tickformat='%d %b',
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
            title_font=dict(color="black"),
            tickfont=dict(family='Arial', size=12, color='black')
        ),
        autosize=True,
        margin=dict(autoexpand=True, l=100, r=100, t=110),
        showlegend=showlegend
    )

    fig.update_xaxes(tickangle=-45)

    return fig

#####################################################
# MAIN PAGE LAYOUT AND INTERACTIVITY
#####################################################

# Layout for inputs
col1, col2, col3, col4 = st.columns([3, 3, 3, 3])

# Input widgets for prediction
with col1:
    date_range = st.date_input('Select Date Range', [datetime.date(2023, 5, 19), datetime.date(2023, 5, 20)])
    selected_model = st.selectbox('Select Prediction Model', ['Prophet', 'Auto-SARIMA'])

with col2:
    is_business = st.selectbox('Is Business?', ['Yes', 'No'])

with col3:
    county_name = st.selectbox('Select County', list(county_name_mapping.values()))

with col4:
    county_number = get_county_number_from_name(county_name)
    product_types = get_available_product_types(county_number, is_business)
    selected_product_type = st.selectbox('Select Product Type', product_types)

# Predict button and results display
if st.button('Predict'):
    if len(date_range) == 2:
        prediction_unit_id = find_pred_id(is_business, county_number, selected_product_type)

        progress_bar = st.empty()
        progress_emojis = ''
        total_length = 42

        # Start the progress bar with a loop
        for i in range(int(30)):  # Start till half-way
            time.sleep(0.025)  # Time delay for each update
            progress_emojis += ':zap:'
            progress_bar.markdown(f'<span style="font-size: 30px;">{progress_emojis}</span>', unsafe_allow_html=True)

        # Fetch forecast data
        forecast_conso, forecast_prod = predict_electricity(date_range[0], date_range[1], prediction_unit_id, selected_model)

        # Complete the rest of the progress bar after data is fetched
        for i in range(30, total_length):
            time.sleep(0.05)
            progress_emojis += ':zap:'
            progress_bar.markdown(f'<span style="font-size: 30px;">{progress_emojis}</span>', unsafe_allow_html=True)

        if forecast_conso is None or forecast_prod is None:
            st.error("🚨🤯🚨🤯🚨🤯🚨🤯🚨🤯🚨🤯")
        else:
            # Filter the actual data based on the prediction unit ID
            filtered_data = y_true[y_true['prediction_unit_id'] == prediction_unit_id]

            # Separate data into consumption and production
            consumption_data = filtered_data[filtered_data['is_consumption'] == 1]
            production_data = filtered_data[filtered_data['is_consumption'] == 0]

            # Create charts for actual vs predicted consumption and production
            fig1 = create_plotly_chart(consumption_data, forecast_conso, 'Graph for Consumption', 'Consumption', showlegend=False)
            fig2 = create_plotly_chart(production_data, forecast_prod, 'Graph for Production', 'Production',showlegend=True)

            col1, col2 = st.columns([6, 6])

            with col1:
                st.plotly_chart(fig1)

            with col2:
                st.plotly_chart(fig2)

    else:
        st.error('Please select both a start and an end date.')
