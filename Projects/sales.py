import pandas as pd
import streamlit as st
import unidecode
import matplotlib.pyplot as plt
import seaborn as sns
import re

# DATA CLEANING FUNCTIONS

def view_nan(df):
    nan_count = df.isna().sum()
    nan_percent = (nan_count / len(df)) * 100
    nan_summary = pd.DataFrame({"NaN Count": nan_count, "NaN (%)": nan_percent})
    nan_summary = nan_summary[nan_summary["NaN Count"] > 0]
    nan_summary = nan_summary.sort_values(by="NaN (%)", ascending=False)
    return nan_summary

def fill_nan(df, col="SALES", method="mean", fixed_value=None):
   
    df = df.copy()
    
    if isinstance(col, str):
        col = [col]
    for c in col:
        if c in df.columns:
            if method in ["mean", "median"] and pd.api.types.is_numeric_dtype(df[c]):
                if method == "mean":
                    df[c] = df[c].fillna(df[c].mean())
                elif method == "median":
                    df[c] = df[c].fillna(df[c].median())
            elif method == "mode":
                if not df[c].mode().empty:
                    df[c] = df[c].fillna(df[c].mode().iloc[0])
            elif method == "fixed" and fixed_value is not None:
                df[c] = df[c].fillna(fixed_value)
    return df

def remove_columns_with_many_nan(df, max_percent=60):
    limit = (max_percent / 100) * len(df)
    mask = df.isna().sum() <= limit
    return df.loc[:, mask]

def padronize_data_types(df):
    expected_types = {
        "ORDERNUMBER": "Int64",
        "QUANTITYORDERED": "Int64",
        "PRICEEACH": "float",
        "ORDERLINENUMBER": "Int64",
        "SALES": "float",
        "ORDERDATE": "datetime64[ns]",
        "STATUS": "category",
        "QTR_ID": "Int64",
        "MONTH_ID": "Int64",
        "YEAR_ID": "Int64",
        "PRODUCTLINE": "category",
        "MSRP": "float",
        "PRODUCTCODE": "string",
        "CUSTOMERNAME": "string",
        "PHONE": "string",
        "ADDRESSLINE1": "string",
        "ADDRESSLINE2": "string",
        "CITY": "string",
        "STATE": "string",
        "POSTALCODE": "string",
        "COUNTRY": "string",
        "TERRITORY": "string",
        "CONTACTLASTNAME": "string",
        "CONTACTFIRSTNAME": "string",
        "DEALSIZE": "category",
    }
    for col, dtype in expected_types.items():
        if col in df.columns:
            try:
                if dtype.lower().startswith("int"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                elif dtype == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                elif dtype == "datetime64[ns]":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                st.error(f"Error trying to convert {col}: {e}")
    return df

def remove_check_duplicates(df, subset=None):
    df = df.copy()
    duplicates_count = df.duplicated(subset=subset).sum()
    if duplicates_count > 0:
        st.warning(f"{duplicates_count} duplicate rows found and removed")
        df = df.drop_duplicates(subset=subset)
    return df

def check_invalid_business_values(df, cols=None):

    df = df.copy()
    if not cols:
        cols = ["QUANTITYORDERED", "PRICEEACH", "SALES"]
    if isinstance(cols, str):
        cols = [cols]
    if "QUANTITYORDERED" in cols and "QUANTITYORDERED" in df.columns:
        neg_qty = (df["QUANTITYORDERED"] < 0).sum()
        if neg_qty > 0:
            st.warning(f"{neg_qty} rows with negative QUANTITYORDERED found and removed")
            df = df[df["QUANTITYORDERED"] > 0]
    if "PRICEEACH" in cols and "PRICEEACH" in df.columns:
        price_zero = (df["PRICEEACH"] <= 0).sum()
        if price_zero > 0:
            st.warning(f"{price_zero} rows with PRICEEACH <= 0 found and removed")
            df = df[df["PRICEEACH"] > 0]
    if "SALES" in cols and "SALES" in df.columns:
        sales_zero = (df["SALES"] <= 0).sum()
        if sales_zero > 0:
            st.warning(f"{sales_zero} rows with SALES <= 0 found and removed")
            df = df[df["SALES"] > 0]
    return df

def check_outliers_zscore(df, cols, limit=3):
   
    df = df.copy()
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            z = (df[col] - df[col].mean()) / df[col].std()
            outliers = df[abs(z) > limit]
            outlier_count = len(outliers)
            if outlier_count > 0:
                st.info(f"{outlier_count} outliers detected in {col} (Z > {limit}) and removed")
                df = df[abs(z) <= limit]
    return df

def check_sales_consistency(df):
    df = df.copy()
    if set(["QUANTITYORDERED", "PRICEEACH", "SALES"]).issubset(df.columns):
        inconsistent = abs(df["QUANTITYORDERED"] * df["PRICEEACH"] - df["SALES"]) > 1e-2
        inconsistency_count = inconsistent.sum()
        if inconsistency_count > 0:
            st.warning(f"{inconsistency_count} rows where SALES != QUANTITYORDERED * PRICEEACH found and removed")
            df = df[~inconsistent]
    return df

def cleaning_strings(df, cols):
    df = df.copy()
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(unidecode.unidecode)
    return df

def check_phone_format(df, cols="PHONE"):
    
    df = df.copy()
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        if col in df.columns:
            invalid = ~df[col].astype(str).str.match(r"^\+?\d[\d -]{7,}$")
            count = invalid.sum()
            if count > 0:
                st.warning(f"{count} invalid phone numbers in {col} removed")
                df = df[~invalid]
    return df

def check_date_consistency(df):
    df = df.copy()
    if set(["ORDERDATE", "YEAR_ID", "MONTH_ID", "QTR_ID"]).issubset(df.columns):
        df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
        invalid_dates = df["ORDERDATE"].isna().sum()
        if invalid_dates > 0:
            st.warning(f"{invalid_dates} rows have invalid ORDERDATE values and were removed")
            df = df[~df["ORDERDATE"].isna()]
        order_year = df["ORDERDATE"].dt.year
        order_month = df["ORDERDATE"].dt.month
        order_quarter = (order_month - 1) // 3 + 1
        year_mismatch = order_year != df["YEAR_ID"]
        month_mismatch = order_month != df["MONTH_ID"]
        qtr_mismatch = order_quarter != df["QTR_ID"]
        if year_mismatch.sum() > 0:
            st.warning(f"{year_mismatch.sum()} rows with YEAR_ID different from ORDERDATE year removed")
            df = df[~year_mismatch]
        if month_mismatch.sum() > 0:
            st.warning(f"{month_mismatch.sum()} rows with MONTH_ID different from ORDERDATE month removed")
            df = df[~month_mismatch]
        if qtr_mismatch.sum() > 0:
            st.warning(f"{qtr_mismatch.sum()} rows with QTR_ID different from ORDERDATE quarter removed")
            df = df[~qtr_mismatch]
    return df




# DATA FUNCTIONS AND PLOTS


# TOP CLIENTS

def plot_top_clients(df, n=5):
    top_clients = df.groupby("CUSTOMERNAME")["SALES"].sum().sort_values(ascending=False).head(n)
    st.write(f"Top {n} clients ")
    fig, ax = plt.subplots()
    sns.barplot(x=top_clients.values, y=top_clients.index, ax=ax)
    ax.set_xlabel("Total sales")
    ax.set_ylabel("Client")
    st.pyplot(fig)


# MONTHS WITH HIGHEST SALES

def plot_top_months(df, n=5):
    if "ORDERDATE" in df.columns:
        df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
        df["YEAR_MONTH"] = df["ORDERDATE"].dt.strftime('%Y-%m')
        top_months = df.groupby("YEAR_MONTH")["SALES"].sum().sort_values(ascending=False).head(n)
        st.write(f"Top {n} months with highest sales")
        fig, ax = plt.subplots()
        sns.barplot(x=top_months.values, y=top_months.index, ax=ax)
        ax.set_xlabel("Total sales")
        ax.set_ylabel("Year-Month")
        st.pyplot(fig)
    else:
        st.warning("Column orderdate not found")


# TOP SELLING PRODUCTS

def plot_top_products(df, n=5):

    top_products = df.groupby("PRODUCTCODE")["QUANTITYORDERED"].sum().sort_values(ascending=False).head(n)
    st.write(f"Top {n} produtos mais vendidos")
    fig, ax = plt.subplots()
    sns.barplot(x=top_products.values, y=top_products.index, ax=ax)
    ax.set_xlabel("Quantity Sold")
    ax.set_ylabel("Product")
    st.pyplot(fig)


# INACTIVE CLIENTS

def plot_inactive_clients(df, reference_month=None):
   
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
    last_date = df["ORDERDATE"].max()
    if reference_month is None:
        reference_month = last_date.strftime("%Y-%m")
    # Clients who bought in the past but remain inactive
    active_clients = set(df[df["ORDERDATE"].dt.strftime("%Y-%m") == reference_month]["CUSTOMERNAME"])
    all_clients = set(df["CUSTOMERNAME"].unique())
    inactive_clients = all_clients - active_clients
    st.write(f"Clients who did not buy in {reference_month}:")
    st.write(list(inactive_clients))


# SALES FOR EACH COUNTRY (TOP SALES)    

def plot_top_countries(df, n=5):
    if "COUNTRY" in df.columns:
        top_countries = df.groupby("COUNTRY")["SALES"].sum().sort_values(ascending=False).head(n)
        st.write(f"Top {n} countries")
        fig, ax = plt.subplots()
        sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax)
        ax.set_xlabel("Total sales")
        ax.set_ylabel("Country")
        st.pyplot(fig)
    else:
        st.warning("Column country not found")


# TRENDING PRODUCTS

def plot_sales_trend(df, freq="M"):
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
    sales = df.groupby(pd.Grouper(key="ORDERDATE", freq=freq))["SALES"].sum().reset_index()
    st.write("Sales tendency throughout the years")
    fig, ax = plt.subplots()
    sns.lineplot(x="ORDERDATE", y="SALES", data=sales, ax=ax, marker="o")
    ax.set_xlabel("Data")
    ax.set_ylabel("Total sales")
    st.pyplot(fig)



# STREAMLIT INTERFACE


st.title(" Data cleaning Pipeline and simple sales analysis ")

# Loading csv file

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file,encoding="latin1")
    st.success("File succesfully loaded")

    # Show nan values

    st.header("Summary of missing values (NaN)")
    st.dataframe(view_nan(df))

    # Asking the user if he wants to apply the functions

    st.header("Data Cleaning options (select the ones you want)")
    df_clean = df.copy()

    # Filling nan

    if st.checkbox("Fill nan values?"):
        col_nan = st.selectbox("Select the column to apply the filling:", df_clean.columns)
        method_nan = st.selectbox("Method:", ["mean", "median", "mode", "fixed"])
        fixed_value = None
        if method_nan == "fixed":
            fixed_value = st.text_input("Fixed value:", value="0")
            try:
                fixed_value = float(fixed_value)
            except Exception:
                pass
        df_clean = fill_nan(df_clean, col=col_nan, method=method_nan, fixed_value=fixed_value)
        st.success(f"NaNs filled in {col_nan} filled by {method_nan}")

    # Removing columns with many nan

    if st.checkbox("Remove columns with many nan?"):
        max_percent = st.slider("max percentual allowed:", 0, 100, 60)
        df_clean = remove_columns_with_many_nan(df_clean, max_percent=max_percent)
        st.success("Columns removed according to the limit")

    # Padronize data types
    if st.checkbox("Standardize data types?"):
        df_clean = padronize_data_types(df_clean)
    st.success("Data types standardized.")

    # Remover duplicatas

    # Remove duplicates
    if st.checkbox("Remove duplicates?"):
        df_clean = remove_check_duplicates(df_clean)
        st.success("Duplicates removed.")

    # Remove invalid business values
    if st.checkbox("Remove invalid business values?"):
        df_clean = check_invalid_business_values(df_clean)
        st.success("Invalid values removed.")

    # Remove outliers
    if st.checkbox("Remove outliers (z-score)?"):
        cols_num = df_clean.select_dtypes(include="number").columns
        col_outlier = st.selectbox("Column for outlier:", cols_num)
        limit = st.slider("Z-score limit", 1, 5, 3)
        df_clean = check_outliers_zscore(df_clean, col_outlier, limit)
        st.success("Outliers removed.")

    # Sales consistency
    if st.checkbox("Remove inconsistent sales rows?"):
        df_clean = check_sales_consistency(df_clean)
        st.success("Inconsistent rows removed.")

    # Clean strings
    if st.checkbox("Clean strings (accents, spaces, uppercase)?"):
        cols_obj = df_clean.select_dtypes(include="object").columns
        cols_str = st.multiselect("Columns for string cleaning:", cols_obj)
        if cols_str:
            df_clean = cleaning_strings(df_clean, cols_str)
            st.success("Strings cleaned.")

    # Phone format
    if st.checkbox("Remove invalid phone numbers?"):
        cols_tel = [c for c in df_clean.columns if "PHONE" in c.upper()]
        cols_tel_select = st.multiselect("Phone columns:", cols_tel)
        if cols_tel_select:
            df_clean = check_phone_format(df_clean, cols_tel_select)
            st.success("Invalid phone numbers removed.")

    # Date consistency
    if st.checkbox("Remove date inconsistencies?"):
        df_clean = check_date_consistency(df_clean)
        st.success("Date consistency checked.")

    #  Show cleaned DataFrame
    st.header("Preview of DataFrame after cleaning")
    st.dataframe(df_clean.head(20))

    # Data visualization with charts
    st.header("Visualizations")

    st.subheader("Top clients (top 5)")
    plot_top_clients(df_clean)

    st.subheader("Months with highest sales (top 5)")
    plot_top_months(df_clean)

    st.subheader("Top selling products (top 5)")
    plot_top_products(df_clean)

    st.subheader("Countries with most purchases (top 5)")
    plot_top_countries(df_clean)

    st.subheader("Clients who stopped buying")
    plot_inactive_clients(df_clean)

    st.subheader("Sales trend over time")
    plot_sales_trend(df_clean)