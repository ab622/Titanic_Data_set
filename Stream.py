import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Prepare data for figures
numeric_df = df.select_dtypes(include='number').dropna()

# Set up the page title and layout
st.set_page_config(page_title='Titanic Dataset Dashboard', layout='wide')

# Background image and title
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(./assets/titanic_background.jpg);
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Titanic Dataset Dashboard")

# Create a row of statistics cards
st.markdown('### Key Statistics')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Passengers", len(df))
col2.metric("Survived", df['Survived'].sum())
col3.metric("Total Fare", f"${df['Fare'].sum():.2f}")
col4.metric("Average Age", f"{df['Age'].mean():.2f}")

# Create a selection box for different plots
st.sidebar.title("Select Analysis Type")
analysis_type = st.sidebar.radio(
    "Choose a plot to display:",
    ('Age Distribution', 'Fare Distribution', 'Survival by Class', 'Survival by Gender', 'Correlation Heatmap', 'Scatter Matrix')
)

# Render different plots based on selection
if analysis_type == 'Age Distribution':
    fig = px.histogram(df, x='Age', nbins=30, title='Distribution of Passenger Ages')
elif analysis_type == 'Fare Distribution':
    fig = px.histogram(df, x='Fare', nbins=30, title='Distribution of Passenger Fares')
elif analysis_type == 'Survival by Class':
    fig = px.histogram(df, x='Pclass', color='Survived', barmode='group', title='Survival Rates by Passenger Class')
elif analysis_type == 'Survival by Gender':
    fig = px.histogram(df, x='Sex', color='Survived', barmode='group', title='Survival Rates by Gender')
elif analysis_type == 'Correlation Heatmap':
    fig = px.imshow(numeric_df.corr(), text_auto=True, aspect='auto', title='Correlation Heatmap')
elif analysis_type == 'Scatter Matrix':
    fig = go.Figure(data=go.Splom(
        dimensions=[{'label': col, 'values': numeric_df[col]} for col in numeric_df.columns if col != 'Survived'],
        showupperhalf=False,
        text=numeric_df['Survived'],
        marker=dict(color=numeric_df['Survived'], colorscale='Viridis', size=5, showscale=False)
    ))
    fig.update_layout(title='Scatter Matrix of Selected Features', dragmode='select')

# Display the plot
st.plotly_chart(fig, use_container_width=True)
