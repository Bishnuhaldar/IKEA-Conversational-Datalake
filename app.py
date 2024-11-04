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
st.markdown('<h1>Conversational Datalake🤖</h1>', unsafe_allow_html=True)

# Path to your logo
logo_path2 = "gcp+capg+ikea.png"

# Display the logo in the sidebar
st.sidebar.image(logo_path2, use_column_width=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input via chat input
user_input = st.chat_input("Ask a question about the data...")





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

questions=['what are the potential options to reduce churn?','what are the options to bring down marketing costs?','key options to increase customer satisfaction','how to increase customer acquisition?','how to reduce acquisition cost?']
answers="""1. What are the potential options to reduce churn?

Churn Analysis Overview

Churn Reason Number of Customers Percentage of Total Churned

Product Quality 500 25%

Poor Customer Service 400 20%

High Costs 300 15%

Lack of Product

Availability 250 12.5%

Delivery Issues 200 10%

Step 1: Identify Key Churn Drivers

The top three reasons for churn account for 60% of total churn:

1. Product Quality (25%)

2. Poor Customer Service (20%)

3. High Costs (15%)

Step 2: Suggested Strategies to Reduce Churn

1. Improve Product Quality

2. Enhance Customer Service

3. Address High Costs





2. What are the options to bring down marketing costs?

Initial Marketing Spend Across Channels:

· Digital Ads: $500,000

· In-Store Promotions: $200,000

· Email Campaigns: $150,000

· Social Media Ads: $100,000

Ways to optimization

· Targeted Campaigns: by focusing on high-value customers.

· Digital Advertising Efficiency: by reallocating budget from low-conversion channels.

· In-Store and Online Integration: unified campaign efforts.

· Social Media Optimization: by leveraging user-generated content.



3. Key options to increase customer satisfaction

Customer Complaints Number of Customers

Product Quality 500

Poor Customer Service 400

Payment Issues 300

Inflexible returns policy 250

Delivery Issues 200



Based on feedback & complaints from customers, the top 5 options to increase customer satisfaction are

Enhance Product Quality

Improve Customer Service

Flexible Payment and Shipping Options

User-Friendly Returns Policy

Streamlined Shipping process



4. How to increase customer acquisition?

Summary of Conversion Rates

Channel Leads New Customers Conversion Rate

Social Media 10,000 1,000 10%

Email Marketing 8,000 800 10%

Paid Advertising 12,000 1,200 10%

Referral Program 5,000 600 12%

Content Marketing 15,000 750 5%


Enhance the referral program and increase the incentives,


5. How to reduce acquisition cost?

Summary of Customer Acquisition Costs

Channel New Customers Marketing Spend ($) CAC ($)

Social Media 1,000 10,000 10

Email Marketing 800 8,000 10

Paid Advertising 1,200 15,000 12.50

Referral Program 600 3,000 5

Content Marketing 750 5,000 6.67

Referral Program has the lowest CAC at $5, while Paid Advertising has the highest CAC at $12.50.

1. Enhance the Referral Program & Increase Incentives:

2. Optimize Paid Advertising & improve targeting

3. Leverage Email Marketing using segmented campaigns & personalization

4. Boost Content Marketing Efficiency by optimizing content for search engines

5. Analyze and Reallocate Budget from paid advertising to referral & content

6. How to increase retention

Metric Value

Total Customers 50,000

Customers Active Last Year 40,000

Customers Retained (1 Year) 30,000

Retention Rate

Average Customer Lifetime (Years) 5

Average Purchase Frequency (Annual) 3

Average Purchase Value $100

Customer Lifetime Value (CLV) 100×3×5=1500

By analyzing the customer segments,

· New Customers (1st Year): 10,000 (Retention Rate: 60%)

· Existing Customers (2+ Years): 30,000 (Retention Rate: 80%)

· High-Value Customers: 5,000 (Retention Rate: 90%)

· Low-Value Customers: 45,000 (Retention Rate: 70%)

Step 3: Strategies to Increase Retention Rate

1. Enhance new Customer Engagement

2. Improve Customer Service

3. Implement high-value customer loyalty program

4. Regular Feedback Collection

5. Personalized Marketing Offersd."""
# Handle user input and query generation
if user_input:
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    if user_input.strip().lower() in questions:
        prompt=f"""a user is asking questions. user questions={user_input}
        
        answer the user on the basis of following data.
        {answers}.
        #########################
        dont add any additional comment just answer the questions if answer having table u can use table with answers.
        use bold for heading and bullet points as well for better representaions of answers.
        """
        
        with st.spinner("Please Wait..."):
            result = qgen(prompt)
            st.write(result)
            
    
    else:
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
if "messages" in st.session_state:
    # Find the last message that contains results
    last_data = None
    for message in reversed(st.session_state.messages):
        if "results" in message:
            last_data = message["results"]
            break
    
    if last_data is not None and not last_data.empty:
        st.write("## Data Visualization")

        numeric_columns = last_data.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_columns = last_data.select_dtypes(exclude=['float64', 'int64']).columns

        for chart_type, selected in chart_types.items():
            if selected:
                st.write(f"### {chart_type}")
                if chart_type == "Bar Chart" and len(numeric_columns) >= 1 and len(non_numeric_columns) >= 1:
                    fig = px.bar(last_data, x=non_numeric_columns[0], y=numeric_columns[0], color=non_numeric_columns[0])
                    st.plotly_chart(fig)
                elif chart_type == "Pie Chart" and len(numeric_columns) >= 1 and len(non_numeric_columns) >= 1:
                    fig = px.pie(last_data, values=numeric_columns[0], names=non_numeric_columns[0])
                    st.plotly_chart(fig)
                elif chart_type == "Line Chart" and len(numeric_columns) >= 1 and len(non_numeric_columns) >= 1:
                    fig = px.line(last_data, x=non_numeric_columns[0], y=numeric_columns[0])
                    st.plotly_chart(fig)
                elif chart_type == "Histogram" and len(numeric_columns) >= 1:
                    fig = px.histogram(last_data, x=numeric_columns[0])
                    st.plotly_chart(fig)
                elif chart_type == "Radar Chart" and len(numeric_columns) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=last_data[numeric_columns].mean().values,
                        theta=numeric_columns,
                        fill='toself'
                    ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
                    st.plotly_chart(fig)
                else:
                    st.warning(f"Not enough appropriate columns to plot a {chart_type}.")
    else:
        st.write("")
