import streamlit as st
import pandas as pd
import requests

with open("api.txt", "r") as f:
    api_key = f.read().strip()
BASE_URL = f"https://mc-api.marketcheck.com/v2/search/car/active?api_key={api_key}"


def fetch_data(year, make, model):
    params = {"year": year, "make": make}
    if model:
        params["model"] = model

    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json().get("listings", [])
    return []


def adj_capcost_to_monthly(adj_cap_cost, months):
    return adj_cap_cost / months  # Simplified formula


def compute_lease_options(
    price, months=36, tax_rate=0.07, bank_fee=500, dmv_fee=100, doc_fee=150
):
    lease_options = {}

    fees_sum = bank_fee + dmv_fee + doc_fee
    adjusted_cap_cost = price
    total_taxes = price * tax_rate

    # 1. TAXES AND FEES UPFRONT
    base_monthly = adj_capcost_to_monthly(adjusted_cap_cost, months)
    lease_options["TAXES AND FEES UPFRONT"] = {
        "Monthly Payment": round(base_monthly, 2),
        "First Payment": round(base_monthly + total_taxes + fees_sum, 2),
    }

    # 2. LEASE TAX INCLUDED
    monthly_lt_included = adj_capcost_to_monthly(
        adjusted_cap_cost + total_taxes, months
    )
    lease_options["LEASE TAX INCLUDED"] = {
        "Monthly Payment": round(monthly_lt_included, 2),
        "First Payment": round(monthly_lt_included + fees_sum, 2),
    }

    return pd.DataFrame.from_dict(lease_options, orient="index")


def main():
    st.set_page_config(page_title="Car Data Dashboard", layout="wide")
    st.title("Car Data Dashboard")

    with st.sidebar:
        st.header("Filter Options")
        year = st.number_input(
            "Year", min_value=2000, max_value=2025, value=2024, step=1
        )
        make = st.text_input("Make", "Ford")
        model = st.text_input("Model", "F-150")
        fetch_button = st.button("Fetch Data")

    if fetch_button:
        with st.spinner("Fetching data..."):
            data = fetch_data(year, make, model)

            if data:
                df = pd.DataFrame(data)
                df_expanded = df["build"].apply(pd.Series)
                df = pd.concat([df.drop(columns=["build"]), df_expanded], axis=1)

                df1 = df[["year", "make", "model", "trim", "price", "miles", "msrp"]]
                selected_rows = st.data_editor(
                    df1, disabled=[], key="table", use_container_width=True
                )

                if selected_rows.get("selected_rows"):
                    selected_index = selected_rows["selected_rows"][0]
                    selected_price = df1.iloc[selected_index]["price"]

                    lease_df = compute_lease_options(selected_price)
                    st.subheader("Lease Computation")
                    st.dataframe(lease_df)

            else:
                st.warning("No data found for the selected filters.")


if __name__ == "__main__":
    main()
