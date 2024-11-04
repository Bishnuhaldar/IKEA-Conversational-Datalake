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

questions=['what are the potential options to reduce churn by 2%','what are the options to bring down marketing costs','key options to increase customer satisfaction','how to increase customer acquisition by 20%','how to reduce acquisition cost','how to increase retention','increase market share & profitability','what was the total sales revenue for the last quarter','can you segment our customers based on their purchase frequency','what are our most frequent purchasers buying that our least frequent purchasers aren’t',
           'are there any other distinguishing factors between the purchase frequency segments','what is the overall sentiment of our customer reviews','can you summarize the key points of feedback from our customer reviews','is there any missing data or other questions that we should be asking for customer feedback on that could improve analysis']
answers="""
#########################################################################

What are the potential options to reduce churn by 2%? 

Churn Analysis Overview 

Churn Reason 

Number of Customers 

Percentage of Total Churned 

Product Quality 

500 

25% 

Poor Customer Service 

400 

20% 

High Costs 

300 

15% 

Lack of Product Availability 

250 

12.5% 

Delivery Issues 

200 

10% 

Identify Key Churn Drivers 

The top three reasons for churn account for 60% of total churn: 

Product Quality (25%) 

Poor Customer Service (20%) 

High Costs (15%) 

 

Suggested Strategies to Reduce Churn 

1. Improve Product Quality 

2. Enhance Customer Service 

3. Address High Costs 


######################################################

What are the options to bring down marketing costs? 

 Initial Marketing Spend Across Channels: 

Digital Ads: $500,000 

In-Store Promotions: $200,000 

Email Campaigns: $150,000 

Social Media Ads: $100,000 

Ways to optimize marketing costs are 

Targeted Campaigns: by focusing on high-value customers. 

Digital Advertising Efficiency: by reallocating budget from low-conversion channels. 

In-Store and Online Integration: unified campaign efforts. 

Social Media Optimization: by leveraging user-generated content. 


##########################################################

Key options to increase customer satisfaction 

 

Customer Complaints 

Number of Customers 

Product Quality 

500 

Poor Customer Service 

400 

Payment Issues 

300 

Inflexible returns policy 

250 

Delivery Issues 

200 

 

 

Based on feedback & complaints from customers, the top 5 options to increase customer satisfaction are 

Enhance Product Quality 

Improve Customer Service 

Flexible Payment and Shipping Options 

User-Friendly Returns Policy 

Streamlined Shipping process 

 

#######################################################

How to increase customer acquisition by 20%? 

Summary of Conversion Rates 

Channel 

Leads 

New Customers 

Conversion Rate 

Social Media 

10,000 

1,000 

10% 

Email Marketing 

8,000 

800 

10% 

Paid Advertising 

12,000 

1,200 

10% 

Referral Program 

5,000 

600 

12% 

Content Marketing 

15,000 

750 

5% 

 

Enhance the referral program and increase the incentives to increase the customer acquisition 

 

##################################################
How to reduce acquisition cost? 

Summary of Customer Acquisition Costs 

Channel 

New Customers 

Marketing Spend ($) 

CAC ($) 

Social Media 

1,000 

10,000 

10 

Email Marketing 

800 

8,000 

10 

Paid Advertising 

1,200 

15,000 

12.50 

Referral Program 

600 

3,000 

5 

Content Marketing 

750 

5,000 

6.67 

Referral Program has the lowest CAC at $5, while Paid Advertising has the highest CAC at $12.50. 

Enhance the Referral Program & Increase Incentives:  

Optimize Paid Advertising & improve targeting 

Leverage Email Marketing using segmented campaigns & personalization 

Boost Content Marketing Efficiency by optimizing content for search engines 

Analyze and Reallocate Budget from paid advertising to referral & content  


############################################################ 

How to increase retention? 

Metric 

Value 

Total Customers 

50,000 

Customers Active Last Year 

40,000 

Customers Retained (1 Year) 

30,000 

Retention Rate 

 

Average Customer Lifetime (Years) 

5 

Average Purchase Frequency (Annual) 

3 

Average Purchase Value 

$100 

Customer Lifetime Value (CLV) 

100×3×5=1500 

By analyzing the customer segments,  

New Customers (1st Year): 10,000 (Retention Rate: 60%) 

Existing Customers (2+ Years): 30,000 (Retention Rate: 80%) 

High-Value Customers: 5,000 (Retention Rate: 90%) 

Low-Value Customers: 45,000 (Retention Rate: 70%) 

Strategies to Increase Retention Rate 

Enhance new Customer Engagement 

Improve Customer Service 

Implement high-value customer loyalty program 

Regular Feedback Collection 

Personalized Marketing Offers 



############################################################ 

Increase market share & profitability 

·  Total Customers: 50,000 

·  Customer Segments: 

New Customers (Last 12 months): 15,000 

Returning Customers (2+ purchases): 25,000 

Churned Customers (Last year): 10,000 

Active Customers: 40,000 

·  Average Order Value (AOV): $50 

·  Annual Revenue: $2,500,000 

·  Average Purchase Frequency: 2 times/year 

·  Customer Lifetime Value (CLV): $100 

·  Market Share: 5% of local retail market 

Strategies to Increase Market Share and Profitability 

1. Customer Acquisition and Segmentation: Targeted Marketing Campaigns -  Focus on the 15,000 new customers and increase acquisition by 20%. 

2.Retention Strategies: Reduce Churn by 2% 

3. Increasing Purchase Frequency: Encourage Existing Customers to Purchase More Often: 

4.Loyalty Programs: Introduce a Loyalty Program 

  

################################################################################

What was the total sales revenue for the last quarter? 

Total Customers: 50,000 

Active Customers: 40,000 

Average Order Value (AOV): $50 

Average Purchase Frequency: 2 times/year 

Quarterly Revenue Calculation 

Since the average purchase frequency is 2 times per year, we can determine the average quarterly purchases: 

Total Sales Revenue for the Last Quarter 

The total sales revenue for the last quarter is $1,000,000. 

  

############################################################## 

Can you segment our customers based on their purchase frequency? 

  

Segment 

Description 

Estimated Size 

Non-Purchasers 

0 Purchases 

10,000 

Light Buyers 

1 Purchase per Year 

10,000 

Moderate Buyers 

2 Purchases per Year 

15,000 

Frequent Buyers 

3-4 Purchases per Year 

10,000 

Highly Engaged Customers 

5+ Purchases per Year 

5,000 

  

#############################################################################

What are our most frequent purchasers buying that our least frequent purchasers aren’t? 

Product Category 

Frequent Purchasers (5,000) 

Least Frequent Purchasers (20,000) 

Core Products 

Multiple brands of essentials (e.g., various cereals, snacks) 

One-off purchase of essentials 

Seasonal Products 

Seasonal apparel, holiday decorations 

Minimal seasonal purchases, if any 

Promotional Bundles 

Engaging with buy-one-get-one offers, multi-pack snacks 

Rarely buy bundles, preferring single items 

High-Value Items 

Electronics, larger household items during promotions 

Typically avoid high-value purchases unless necessary 

  
  

##################################################

Are there any other distinguishing factors between the purchase frequency segments? 

Factor 

Frequent Purchasers 

Least Frequent Purchasers 

Demographics 

Younger, families, higher income 

Older, varied income levels 

Shopping Channel 

Prefer online shopping 

Favor in-store shopping 

Behavioral Patterns 

Highly responsive to promotions 

Less responsive to promotions 

Customer Engagement 

Higher email engagement 

Lower email engagement 

Lifetime Value 

Higher CLV, lower churn rates 

Lower CLV, higher churn rates 

Brand Loyalty 

Strong brand affinity 

More brand experimentation 

 Seasonal Trends 

Actively participate in seasonal sales 

Limited participation in seasonal sales 

  


 ###############################################################

What is the overall sentiment of our customer reviews? 

  

·  Positive Reviews: 700 (70%) 

·  Neutral Reviews: 200 (20%) 

·  Negative Reviews: 100 (10%) 

Overall Sentiment score = 0.6 which indicates positive sentiment 

  

#################################################################

Can you summarize the key points of feedback from our customer reviews? 

Key Points of Customer Feedback 

Positive Feedback 

Product Quality 

Many customers praised the quality of the products, noting that they met or exceeded expectations. High-quality essentials and seasonal items were frequently highlighted. 

Customer Service 

Positive comments about responsive and helpful customer service representatives. Customers appreciated quick resolutions to issues and friendly interactions. 

User Experience 

Customers found the online shopping experience easy to navigate, with a user-friendly interface and efficient checkout process. 

Promotions and Discounts 

Customers enjoyed the value offered through promotions, particularly buy-one-get-one deals and loyalty rewards. This incentivized them to make additional purchases. 

Fast Delivery 

Many reviews mentioned satisfaction with quick shipping times, which enhanced the overall shopping experience. 

Neutral Feedback 

Product Variety 

Some customers noted the variety of products available but expressed a desire for more options in specific categories, particularly seasonal or specialty items. 

Pricing Concerns 

Neutral comments indicated that while some products were seen as fairly priced, others were perceived as slightly higher than competitors, but customers appreciated the quality. 

Website Features 

Feedback on website features was mixed. While many found the site easy to use, some suggested improvements for better filtering options and product search capabilities. 

Negative Feedback 

Stock Availability 

Several customers expressed frustration about out-of-stock items, particularly during peak shopping seasons. This led to disappointment and potential loss of sales. 

Return Process 

Negative comments highlighted challenges with the return process, including delays and lack of clarity on return policies, which affected customer satisfaction. 

Inconsistent Product Information 

Some customers reported inconsistencies in product descriptions and images compared to the actual items received, leading to dissatisfaction. 

Delivery Issues 

A few reviews mentioned issues with shipping delays or damaged products upon arrival, impacting overall trust in the service. 

  

##################################################################################################################

   

Is there any missing data or other questions that we should be asking for customer feedback on that could improve analysis? 

Additional Questions for Customer Feedback 

Delivery Experience 

 "How would you rate your delivery experience, and what improvements would you suggest?" This can provide specific insights into logistics and fulfillment. 

Interest in New Products 

 "What types of products would you like to see more of in our store?" This helps in inventory planning and product development. 

Feedback on Marketing 

 "How do you prefer to receive promotions and updates from us?" Understanding communication preferences can help optimize marketing strategies. 

Return Experience 

"How would you rate your experience with returns or exchanges?" This is critical for identifying pain points in the return process. 

  

 
"""
# Handle user input and query generation
if user_input:
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    mock_question=False
    for i in questions:
        
        if i in user_input.strip().lower() :
            mock_question=True
            prompt=f"""a user is asking questions. user questions={user_input}
            
            answer the user on the basis of following following question answer. write full answer of question {user_input} as it is with data(table).dont make any changes in answer.
            {answers}.
            #########################
            dont add any additional comment just answer the questions if answer having table u can use table with answers.
            use bold for heading and bullet points as well for better representaions of answers.
            """
            
            with st.spinner("Please Wait..."):
                result = qgen(prompt)
                st.session_state.messages.append({"role": "assistant", "content":result, "summary": ''})

        
    if mock_question==False:
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
            if not message["results"].empty: 
                st.dataframe(message["results"])
        # if "summary" in message:
        #     st.write(message["summary"])



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
                st.write("## Data Visualization")
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
