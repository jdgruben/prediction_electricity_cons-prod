import streamlit as st
import pandas as pd
import json
import datetime
import time
import plotly.graph_objects as go

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

def predict_electricity(start_date, end_date, prediction_unit_id):
    """Function to predict electricity consumption and production."""
    pass

#####################################################
# PLOTTING FUNCTIONS
#####################################################

def create_plotly_chart(y_true, y_pred, title, y_label):
    """Create a Plotly chart for visualizing actual and predicted values."""
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

#####################################################
# MAIN PAGE LAYOUT AND INTERACTIVITY
#####################################################

# Layout for inputs
col1, col2, col3, col4 = st.columns([3, 3, 3, 3])

# Input widgets for prediction
with col1:
    date_range = st.date_input('Select Date Range', [datetime.date(2023, 5, 19), datetime.date(2023, 5, 20)])

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
        st.write(f'Prediction Unit ID: {prediction_unit_id}')

        filtered_data = y_true[y_true['prediction_unit_id'] == prediction_unit_id]

        consumption_data = filtered_data[filtered_data['is_consumption'] == 1]
        production_data = filtered_data[filtered_data['is_consumption'] == 0]

        fig1 = create_plotly_chart(consumption_data, None, 'Graph for Consumption', 'Consumption')
        fig2 = create_plotly_chart(production_data, None, 'Graph for Production', 'Production')

        progress_bar = st.empty()
        progress_emojis = ''
        total_length = 42

        for i in range(total_length):
            time.sleep(0.04)
            progress_emojis += ':zap:'
            progress_bar.markdown(f'<span style="font-size: 30px;">{progress_emojis}</span>', unsafe_allow_html=True)

        col1, col2 = st.columns([6, 6])

        with col1:
            st.plotly_chart(fig1)

        with col2:
            st.plotly_chart(fig2)

    else:
        st.error('Please select both a start and an end date.')
