import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

st.title("🛒 Retail Sales Analysis Dashboard")
st.write("Upload your retail dataset CSV file to begin the analysis.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    # Convert InvoiceDate
    try:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    except:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M")

    # Basic Cleaning
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # remove cancelled orders
    df = df.dropna(subset=["CustomerID"])  # remove missing customers
    df = df[df["UnitPrice"] > 0]  # remove wrong unit prices
    df = df.drop_duplicates()

    # Total Revenue
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    st.subheader("📊 Sample Data")
    st.dataframe(df.head())

    # Feature Extraction
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")

    country = st.sidebar.selectbox("Select Country", ["All"] + sorted(df["Country"].unique()))
    if country != "All":
        df = df[df["Country"] == country]

    # Revenue by StockCode
    top_rev = df.groupby("StockCode")["TotalPrice"].sum().sort_values(ascending=False).head(10)

    st.subheader("🏆 Top 10 Revenue-Generating Stock Codes")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    top_rev.plot(kind="bar", ax=ax1)
    ax1.set_title("Top 10 Products by Revenue")
    ax1.set_ylabel("Revenue")
    st.pyplot(fig1)

    # Monthly Sales Trend
    monthly_sales = df.groupby(["Year", "Month"])["TotalPrice"].sum()

    st.subheader("📈 Monthly Sales Trend")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    monthly_sales.plot(ax=ax2)
    ax2.set_title("Monthly Revenue Over Time")
    ax2.set_ylabel("Revenue")
    st.pyplot(fig2)

    # Country Sales Pie Chart
    st.subheader("🌍 Revenue Contribution by Country")
    country_sales = df.groupby("Country")["TotalPrice"].sum()

    fig3, ax3 = plt.subplots(figsize=(8, 8))
    country_sales.plot(kind="pie", autopct="%1.1f%%", ax=ax3)
    ax3.set_ylabel("")
    st.pyplot(fig3)

else:
    st.info("Please upload a dataset to proceed.")
