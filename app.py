import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import google.generativeai as genai
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure the OpenAI API
key = "AIzaSyB2Ap-o973pkpyvPaKiktbZwd4LX1FxU2c"
genai.configure(api_key=key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}
# Fetch the service account key from Streamlit secrets
service_account_info = st.secrets["GCP_SERVICE_ACCOUNT_KEY"]
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# # Load credentials from service account file
# credentials = service_account.Credentials.from_service_account_file('data-driven-cx.json')

# Set up BigQuery client
project_id = 'data-driven-cx'
client = bigquery.Client(credentials=credentials, project=project_id)

# Custom CSS for modern look and feel
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        body {
            font-family: 'Roboto', sans-serif;
            color: #333333;
            background-color: #F0F2F6;
        }
        
        h1 {
            font-weight: bold;
            color: #1E90FF;
        }
        
        .sidebar .sidebar-content {
            background-color: #FFFFFF;
        }
        
        .stTextInput > div {
            background-color: #FFFFFF;
            border-radius: 5px;
            border: 1px solid #1E90FF;
            padding: 10px;
            margin-top: 10px;
        }
        
        button {
            background-color: #1E90FF;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        
        button:hover {
            background-color: #4682B4;
        }
        
        .stSlider {
            padding: 10px 0;
        }
        
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
        }
        
        .card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* Dark mode styles */
        .dark-mode {
            color: #F0F2F6;
            background-color: #1E1E1E;
        }
        
        .dark-mode .sidebar .sidebar-content {
            background-color: #2D2D2D;
        }
        
        .dark-mode .stTextInput > div {
            background-color: #2D2D2D;
            border-color: #4682B4;
        }
        
        .dark-mode .card {
            background-color: #2D2D2D;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1>IKEA Conversational Datalake🤖</h1>', unsafe_allow_html=True)

# Path to your logo
logo_path2 = "gcp+capg+ikea.png"

# Display the logo in the sidebar
st.sidebar.image(logo_path2, use_column_width=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input via chat input
user_input = st.chat_input("Ask a question about the data...")

st.sidebar.title("Options")



# Set your project ID and dataset ID
project_id = "data-driven-cx"
dataset_id = "EDW_ECOM"

# Function to fetch table schemas
def fetch_table_schemas(project_id, dataset_id):
    dataset_ref = client.dataset(dataset_id)
    tables = client.list_tables(dataset_ref)

    all_schemas_info = ""
    for table in tables:
        table_ref = dataset_ref.table(table.table_id)
        try:
            table = client.get_table(table_ref)
            schema_str = f"Schema for table {table.table_id}:\n"
            for field in table.schema:
                schema_str += f"  {field.name} ({field.field_type})\n"
            all_schemas_info += schema_str + "\n"
        except Exception as e:
            st.error(f"Table {table.table_id} not found.")
    
    return all_schemas_info

# Fetch and store schema information in session state
if "schema" not in st.session_state:
    st.session_state.schema = []

if st.session_state.schema == []:
    with st.spinner("Loading schema information..."):
        schema_for_tables = fetch_table_schemas(project_id, dataset_id)
        st.session_state.schema.append(schema_for_tables)

# Function to execute SQL queries
def execute_query(query):
    try:
        query_job = client.query(query)
        results = query_job.result().to_dataframe()
        return results
    except Exception as e:
        st.error(f"Query execution error: {e}")
        return None

# Query generation function
def qgen(prompt):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )
    response = model.generate_content(prompt)
    return response.text

# Capture unique COUNTRY_SOURCE_IDs for selection
country_query = f"""
    SELECT DISTINCT customer_state
    FROM `{project_id}.{dataset_id}.olist_customers_datasets`;
"""

# country_data = execute_query(country_query)
# if country_data is not None:
#     countries = country_data['customer_state'].tolist()
#     selected_state = st.sidebar.selectbox("Choose a customer state", countries)

# Slider to set limit of result

# Sidebar for chart type selection
st.sidebar.subheader("Chart Type")
chart_types = {
    "Bar Chart": st.sidebar.checkbox("Bar Chart"),
    "Pie Chart": st.sidebar.checkbox("Pie Chart"),
    "Line Chart": st.sidebar.checkbox("Line Chart"),
    "Histogram": st.sidebar.checkbox("Histogram"),
    "Radar Chart": st.sidebar.checkbox("Radar Chart")
}

limit = st.sidebar.slider('Limit Of Output', 0, 100, 10)

# Handle user input and query generation
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    my_prompt = f"""act as a sql query writer for BigQuery database. We have the following schema:
    project_id = "data-driven-cx"
    dataset_id = "EDW_ECOM"
       {st.session_state.schema[0]}
    Write a SQL query for user input
    user input-{user_input}.
    set limit to {limit}.
    Write only the executable query without any comments or additional text.
    """
    
    with st.spinner("Generating query..."):
        final_query = qgen(my_prompt)
        cleaned_query = final_query.replace("```sql", "").replace("```", "").strip()
    
    try:
        with st.spinner("Executing query..."):
            data = execute_query(cleaned_query)
        st.session_state.messages.append({"role": "assistant", "content": final_query, "results": data})
    except Exception as e:
        st.error(f"Query execution error: {e}")
        if "editable_sql" not in st.session_state:
            st.session_state.editable_sql = final_query

# Display the SQL query editor and execution button if there's a query to edit
if "editable_sql" in st.session_state:
    st.write("## Edit and Re-Execute the Query")
    edited_sql = st.text_area("Edit Query", st.session_state.editable_sql)
    
    if st.button('Submit'):
        with st.spinner("Executing query..."):
            data = execute_query(edited_sql)
        if data is not None:
            st.session_state.messages.append({"role": "assistant", "content": edited_sql, "results": data})
            st.session_state.editable_sql = edited_sql

# Display all the chat messages
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='card'>{message['content']}</div>", unsafe_allow_html=True)
        if "results" in message:
            st.dataframe(message["results"])



# Visualization section
if 'data' in locals() and not data.empty:
    st.write("## Data Visualization")

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_columns = data.select_dtypes(exclude=['float64', 'int64']).columns

    for chart_type, selected in chart_types.items():
        if selected:
            st.write(f"### {chart_type}")
            if chart_type == "Bar Chart" and len(numeric_columns) >= 1 and len(non_numeric_columns) >= 1:
                fig = px.bar(data, x=non_numeric_columns[0], y=numeric_columns[0], color=non_numeric_columns[0])
                st.plotly_chart(fig)
            elif chart_type == "Pie Chart" and len(numeric_columns) >= 1 and len(non_numeric_columns) >= 1:
                fig = px.pie(data, values=numeric_columns[0], names=non_numeric_columns[0])
                st.plotly_chart(fig)
            elif chart_type == "Line Chart" and len(numeric_columns) >= 1 and len(non_numeric_columns) >= 1:
                fig = px.line(data, x=non_numeric_columns[0], y=numeric_columns[0])
                st.plotly_chart(fig)
            elif chart_type == "Histogram" and len(numeric_columns) >= 1:
                fig = px.histogram(data, x=numeric_columns[0])
                st.plotly_chart(fig)
            elif chart_type == "Radar Chart" and len(numeric_columns) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=data[numeric_columns].mean().values,
                    theta=numeric_columns,
                    fill='toself'
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig)
            else:
                st.warning(f"Not enough appropriate columns to plot a {chart_type}.")

