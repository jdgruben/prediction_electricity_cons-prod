import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import json
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------------------
#                           Set page configuration
# ------------------------------------------------------------------------------------------

st.set_page_config(page_title='Electricity Prediction', layout='wide', page_icon=':zap:')
st.markdown("""
    <style>
    body {
        font-family: 'Helvetica', sans-serif;
        text-align: center;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    .main .block-container {
        max-width: 800px; /* Adjust the max width of the central block */
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True)

st.title(':zap: Electricity Consumption and Production Prediction :zap:')

# ------------------------------------------------------------------------------------------
#                           Data Loading Functions
# ------------------------------------------------------------------------------------------

def load_pred_id_mapping():
    """Load prediction unit ID mapping from a JSON file."""
    with open('streamlit/data/mapping_prediction_unit_id.json', 'r') as file:
        pred_id_mapping = json.load(file)
    return pd.DataFrame.from_dict(pred_id_mapping, orient='index')

def load_y_true():
    """Load true data from a CSV file."""
    y_true = pd.read_csv('streamlit/data/y_true.csv')
    y_true['datetime'] = pd.to_datetime(y_true['datetime'])
    return y_true

# ------------------------------------------------------------------------------------------
#                               Helper Functions
# ------------------------------------------------------------------------------------------

def get_county_number_from_name(county_name, county_name_mapping):
    """Get county number from name."""
    return county_name_mapping.get(county_name)

def get_product_type_number(selected_product_name, product_type_mapping):
    """Get product type number from name."""
    return product_type_mapping.get(selected_product_name)

def find_pred_id(is_business, county_number, selected_product_type, pred_id_mapping):
    """Find prediction unit ID."""
    is_business_flag = 1 if is_business == 'Yes' else 0
    product_type_number = get_product_type_number(selected_product_type)
    for _, row in pred_id_mapping.iterrows():
        if row['county'] == int(county_number) and row['is_business'] == is_business_flag and row['product_type'] == int(product_type_number):
            return row['prediction_unit_id']
    return None

def get_available_product_types(county_number, is_business, pred_id_mapping):
    """Get available product types based on county number and business status."""
    is_business_flag = 1 if is_business == 'Yes' else 0
    filtered_df = pred_id_mapping[(pred_id_mapping['county'] == int(county_number)) & (pred_id_mapping['is_business'] == is_business_flag)]
    return [product_type_mapping[str(num)] for num in filtered_df['product_type'].unique().tolist()]

# ------------------------------------------------------------------------------------------
#                               Plotting Function
# ------------------------------------------------------------------------------------------

# Initialize a plotly graph object figure
def create_plotly_chart(y_true, y_pred, title, y_label):
    """Create a Plotly chart with true and predicted data."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true['datetime'], 
        y=y_true['target'],
        mode='lines',
        name='Actual',
        line=dict(color='green')  
    ))
 
    if y_pred is not None:
        fig.add_trace(go.Scatter(
            x=y_pred['datetime'], 
            y=y_pred['target'],
            mode='lines',
            name='Predicted',
            line=dict(color='orange')  
        ))

  
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        plot_bgcolor="#0D1116",  
        paper_bgcolor="#0D1116",  
        font=dict(color="white"),
        title_font=dict(size=20, color="white"),
        title_x=0.35,
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor="white",
            linewidth=2,
            ticks='outside',
            tickfont=dict(family='Arial', size=12, color='white'),
            tickmode='auto',
            tickformat='%d %b',
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
        ),
        autosize=True,
        margin=dict(autoexpand=True, l=100, r=100, t=110),
        showlegend=True
    )

    fig.update_xaxes(tickangle=-45)

    return fig

# ------------------------------------------------------------------------------------------
#                                Main App Layout
# ------------------------------------------------------------------------------------------

# Load data
pred_id_mapping = load_pred_id_mapping()
y_true = load_y_true()

# County and product type mappings
county_name_mapping = {
    "0": "Harjumaa", "1": "Hiiumaa", "2": "Ida-Virumaa", "3": "Järvamaa",
    "4": "Jõgevamaa", "5": "Lääne-Virumaa", "6": "Läänemaa", "7": "Pärnumaa",
    "8": "Põlvamaa", "9": "Raplamaa", "10": "Saaremaa", "11": "Tartumaa",
    "13": "Valgamaa", "14": "Viljandimaa", "15": "Võrumaa"
}
product_type_mapping = {'0': "Combined", '1': "Fixed", '2': "General service", '3': "Spot"}

# Input widgets for predictions
col1, col2, col3, col4 = st.columns([3, 3, 3, 3])

with col1:
    date_range = st.date_input('Select Date Range', [datetime.date(2023, 5, 19), datetime.date(2023, 5, 20)])
with col2:
    is_business = st.selectbox('Is Business?', ['Yes', 'No'])

with col3:
    # Dropdown for county selection
    county_name = st.selectbox('Select County', list(county_name_mapping.values()))
    
with col4:
    county_number = get_county_number_from_name(county_name)
    product_types = get_available_product_types(county_number, is_business)
    selected_product_type = st.selectbox('Select Product Type', product_types) # Update product types based on selected county and business status


# ------------------------------------------------------------------------------------------
#                           Prediction and Visualization Logic
# ------------------------------------------------------------------------------------------

if st.button('Predict'):
    if len(date_range) == 2:
        prediction_unit_id = find_pred_id(is_business, county_number, selected_product_type, pred_id_mapping)
        st.write(f'Prediction Unit ID: {prediction_unit_id}')

        # Filter y_true for the selected prediction_unit_id
        filtered_data = y_true[y_true['prediction_unit_id'] == prediction_unit_id]

        # Split data into consumption and production
        consumption_data = filtered_data[filtered_data['is_consumption'] == 1]
        production_data = filtered_data[filtered_data['is_consumption'] == 0]

        # Pre-generate graphs
        fig1 = create_plotly_chart(consumption_data, None, 'Graph for Consumption', 'Consumption')
        fig2 = create_plotly_chart(production_data, None, 'Graph for Production', 'Production')

        # Initialize progress bar
        progress_bar = st.empty()
        progress_emojis = ''
        total_length = 42  

        for i in range(total_length):
            time.sleep(0.04)  
            progress_emojis += ':zap:'  
            progress_bar.markdown(f'<span style="font-size: 30px;">{progress_emojis}</span>', unsafe_allow_html=True)

        # Display graphs after progress bar completes
        col1, col2 = st.columns([6, 6])

        with col1:
            st.plotly_chart(fig1)

        with col2:
            st.plotly_chart(fig2)

    else:
        st.error('Please select both a start and an end date.')


