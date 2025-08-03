import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain


# Load Groq API Key
def load_api_key():
    load_dotenv()
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = "your api key"
    return os.getenv("GROQ_API_KEY")

# Display welcome message and sidebar
def display_welcome():
    st.set_page_config(page_title="Smart EDA Assistant", layout="wide")
    st.title("Smart EDA Assistant 📊")
    st.write("Hi there! 👋 I'm your AI-powered assistant to help you explore, understand, and model your data efficiently.")

    with st.sidebar:
        st.header("📁 Upload & Explore")
        st.markdown("""
        Start by uploading a CSV file.  
        I'll help you:
        - Understand your dataset
        - Visualize key patterns
        - Frame your business problem
        - Suggest ML models and code
        """)
        st.divider()
        st.caption("Crafted by Akash 💡")

# Handle button click
def clicked(button):
    st.session_state.clicked[button] = True 

# Handle file upload
def handle_file_upload():
    user_csv = st.file_uploader("Upload your CSV file", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        try:
            df = pd.read_csv(user_csv)
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    return None

# Suggestion model using Groq Gemini
def suggestion_model(api_key, topic):
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
    prompt = PromptTemplate.from_template("You are a genius data scientist. Write me a solution {topic}.")
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain.run(topic)

# Display data overview
def data_overview(df, _pandas_agent):
    st.subheader("📌 Dataset Overview")
    st.write("**First few rows:**")
    st.write(df.head())

    st.write("**Column Descriptions:**")
    st.write(_pandas_agent.run("What are the meaning of the columns?"))

    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    st.write("**Duplicate Values:**")
    st.write(_pandas_agent.run("Are there any duplicate values and if so where?"))

    st.write("**Summary Statistics:**")
    st.write(df.describe())

    st.write("**Shape:**")
    st.write(f"{df.shape[0]} rows and {df.shape[1]} columns")

    st.write("**Skewness:**")
    for col in df.select_dtypes(include='number').columns:
        st.write(f"{col}: {df[col].skew()}")

# Display variable insights
def variable_info(df, var):
    st.subheader("📈 Variable Insights")
    st.write(f"**Summary Statistics for '{var}':**")
    st.write(df[var].describe())

    st.write("**Line chart:**")
    st.line_chart(df[var])

    st.write("**histplot:**")
    fig, ax = plt.subplots()
    sns.histplot(df[var], kde=True, ax=ax, color='#4CAF50')
    st.pyplot(fig)

    st.write("**boxplot:**")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[var], ax=ax, color='#4CAF50')
    st.pyplot(fig)

    if df[var].dtype == 'O':
        st.write("**Value Counts:**")
        st.write(df[var].value_counts())
    else:
        if df[var].isnull().all():
            st.write("No data available to detect outliers.")
        else:
            # Compute z-scores only for non-null values
            z_scores = stats.zscore(df[var].dropna())
            z_scores_series = pd.Series(z_scores, index=df[var].dropna().index)
    
            # Create a full-length boolean mask
            outlier_mask = pd.Series(False, index=df.index)
            outlier_mask.loc[z_scores_series.index] = (z_scores_series > 3) | (z_scores_series < -3)
    
            # Extract outliers
            outliers = df.loc[outlier_mask, var]
            st.write("**Outliers:**")
            st.write(outliers)
    
            # Normality test
            try:
                _, p_value = stats.normaltest(df[var].dropna())
                st.write(f"**Normality Test P-value:** {p_value}")
                st.write("✅ Normal distribution" if p_value >= 0.05 else "❌ Not normal distribution")
            except Exception as e:
                st.write(f"Normality test failed: {e}")

    st.write(f"**Missing Values:** {df[var].isnull().sum()}")
    st.write(f"**Data Type:** {df[var].dtype}")

# Perform custom pandas task
def perform_pandas_task(task, _pandas_agent):
    if task:
        return _pandas_agent.run(task)
    return "No task provided."

# Main function
def main():
    api_key = load_api_key()
    display_welcome()

    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1: False}
    st.button("🚀 Start Exploring", on_click=clicked, args=[1])

    if st.session_state.clicked[1]:
        df = handle_file_upload()
        if df is not None:
            llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
            pandas_agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, handle_parsing_errors=True, allow_dangerous_code=True
            )

            st.header("🔍 Exploratory Data Analysis")
            data_overview(df, pandas_agent)

            selected_var = st.selectbox("Select a variable to study", df.columns)
            if selected_var:
                variable_info(df, selected_var)

            st.subheader("🧪 Custom Analysis")
            task_input = st.text_input("What task do you want to perform?")
            if task_input:
                result = perform_pandas_task(task_input, pandas_agent)
                st.write(f"**Result of '{task_input}':**")
                st.write(result)

            with st.sidebar:
                with st.expander("📚 What are the steps of EDA?"):
                    topic = 'What are the steps of Exploratory Data Analysis'
                    st.write(suggestion_model(api_key, topic))

                with st.expander("💬 Ask a Data Science Question"):
                    llm_suggestion = st.text_area("Ask me anything:")
                    if st.button("Get Answer"):
                        st.write(suggestion_model(api_key, llm_suggestion))

if __name__ == '__main__':
    main()
