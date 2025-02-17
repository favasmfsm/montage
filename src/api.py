import requests
import pandas as pd
import streamlit as st


@st.cache_data
def fetch_car_data(api_key, year, make, model, zip_code, radius, msrp_range):
    """Fetch car data from MarketCheck API and return as a DataFrame."""
    BASE_URL = "https://mc-api.marketcheck.com/v2/search/car/active"

    params = {
        "api_key": api_key,
        "year": year,
        "make": make.lower() if make else None,
        "model": model.lower() if model else None,
        "zip": zip_code,
        "radius": radius,
        "msrp_range": msrp_range,
    }

    # Remove None values to avoid API errors
    params = {k: v for k, v in params.items() if v is not None}

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json().get("listings", [])
        return pd.DataFrame(data) if data else None
    except requests.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return None
