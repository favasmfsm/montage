import streamlit as st
import pandas as pd
from utils import parse_float


def sidebar_filters(df):
    """Generates sidebar filters and returns user-selected filter values."""

    st.sidebar.header("Filter Options")
    api_key = st.sidebar.text_input("Enter your API Key", type="password")

    # Lease Term Selection
    lease_terms = st.sidebar.multiselect(
        "Select Lease Term (months)",
        options=[24, 27, 30, 33, 36, 39, 42, 48],
        default=36,
    )

    # Lease Type Selection
    lease_type = st.sidebar.selectbox(
        "Select Lease Type",
        [
            "TAXES AND FEES UPFRONT",
            "LEASE TAX INCLUDED",
            "TAXES AND BANK",
            "First Due",
            "SIGN AND DRIVE $0 DAS",
            "ONEPAY",
        ],
    )

    # Filter by Make
    make_options = sorted(df["Make"].dropna().unique())
    selected_makes = st.sidebar.multiselect(
        "Select Make", options=make_options, default=[]
    )

    # Filter by Year
    year_options = sorted(df["Year"].dropna().unique())
    selected_years = st.sidebar.multiselect(
        "Select Year", options=year_options, default=[]
    )

    # Model Filter
    temp_df = df.copy()
    if selected_makes:
        temp_df = temp_df[temp_df["Make"].isin(selected_makes)]
    if selected_years:
        temp_df = temp_df[temp_df["Year"].isin(selected_years)]

    model_options = sorted(temp_df["Model"].dropna().unique())
    selected_models = st.sidebar.multiselect(
        "Select Model", options=model_options, default=[]
    )

    # Trim Filter
    trim_options = sorted(temp_df["Trim"].dropna().unique())
    selected_trims = st.sidebar.multiselect(
        "Select Trim", options=trim_options, default=[]
    )

    # MSRP Range Filter
    min_price = int(df["MSRP"].min())
    max_price = int(df["MSRP"].max())
    selected_price_range = st.sidebar.slider(
        "Select MSRP Range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
    )

    # Tax & Fees Input Fields
    with st.sidebar.expander("Taxes and Fees Options"):
        tax_rate = parse_float(st.text_input("% Tax Rate", value="8.875"), 8.875) * 0.01
        dmv_fee = parse_float(st.text_input("DMV Fee", value="350"), 350)
        doc_fee = parse_float(st.text_input("Documentation Fee", value="249"), 249)
        bank_fee = parse_float(st.text_input("Bank Fee", value="595"), 595)

    return (
        api_key,
        lease_terms,
        lease_type,
        selected_makes,
        selected_years,
        selected_models,
        selected_trims,
        selected_price_range,
        tax_rate,
        dmv_fee,
        doc_fee,
        bank_fee,
    )
