import streamlit as st
import pandas as pd
import requests

# -----------------------------
# Page & Logo/Banner Setup
# -----------------------------
st.set_page_config(
    page_title="Montage Auto Leasing Dashboard", page_icon=":car:", layout="wide"
)
st.image("src/banner.png", use_container_width=True)
st.sidebar.image("src/logo.png", width=150)

# -----------------------------
# Load CSV Data & Setup Sidebar Filters
# -----------------------------
# Load your merged CSV file
try:
    df = pd.read_csv("./data/merged_data_v3.csv")
except Exception as e:
    st.error("Error loading CSV file.")
    st.stop()

st.sidebar.header("Filter Options")
api_key = st.sidebar.text_input("Enter your API Key", type="password")

# -- CSV Data Filters (using multiselect for multiple selection) --
# Lease term selection (in months)
lease_term = st.sidebar.selectbox(
    "Select Lease Term (months)", options=[24, 27, 30, 33, 36, 39, 42, 48], index=4
)  # default is 36 months

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
selected_makes = st.sidebar.multiselect("Select Make", options=make_options, default=[])

# Filter by Year
year_options = sorted(df["Year"].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year", options=year_options, default=[])

# Apply Make/Year filters if selected
temp_df = df.copy()
if selected_makes:
    temp_df = temp_df[temp_df["Make"].isin(selected_makes)]
if selected_years:
    temp_df = temp_df[temp_df["Year"].isin(selected_years)]

# Model filter (Show all initially)
model_options = sorted(temp_df["Model"].dropna().unique())
selected_models = st.sidebar.multiselect(
    "Select Model", options=model_options, default=[]
)

# Apply Model filter if selected
if selected_models:
    temp_df = temp_df[temp_df["Model"].isin(selected_models)]

# Trim filter (Show all initially)
trim_options = sorted(temp_df["Trim"].dropna().unique())
selected_trims = st.sidebar.multiselect("Select Trim", options=trim_options, default=[])

# Apply Trim filter if selected
if selected_trims:
    temp_df = temp_df[temp_df["Trim"].isin(selected_trims)]

# Filter by MSRP Range (assumes the CSV has an 'MSRP' column)
min_price = int(df["MSRP"].min())
max_price = int(df["MSRP"].max())
selected_price_range = st.sidebar.slider(
    "Select MSRP Range",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
)


with st.sidebar.expander("Taxes and Fees Options"):
    tax_rate = st.text_input("% Tax Rate", value="8.875")
    dmv_fee = st.text_input("DMV Fee", value="350")
    doc_fee = st.text_input("Documentation Fee", value="249")
    bank_fee = st.text_input("Bank Fee", value="595")


# Convert inputs to float while handling empty or invalid values
def parse_float(value, default):
    try:
        return float(value)
    except ValueError:
        return default


tax_rate = parse_float(tax_rate, 8.875) * 0.01  # Convert percentage to decimal
dmv_fee = parse_float(dmv_fee, 350)
doc_fee = parse_float(doc_fee, 249)
bank_fee = parse_float(bank_fee, 595)
# -----------------------------
# Apply Filters to CSV Data
# -----------------------------
filtered_data = df.copy()


# Lease Computation for Each Entry
def compute_lease(row):
    adjusted_cap_cost = row.get("Adjusted Cap Cost", 0)
    residual_value = row.get(f"residual_value_{lease_term}", 0)
    money_factor = row.get(f"MF {lease_term}", 0)
    depreciation_fee = (adjusted_cap_cost - residual_value) / lease_term
    fees_sum = dmv_fee + doc_fee
    total_taxes = depreciation_fee * tax_rate * lease_term

    tfu_net_cap_cost = residual_value + adjusted_cap_cost
    base_monthly = depreciation_fee + (tfu_net_cap_cost * money_factor)
    total_taxes = base_monthly * tax_rate * lease_term
    option1_first = round(base_monthly + total_taxes + fees_sum + bank_fee, 2)
    # st.write(
    #     tfu_net_cap_cost,
    #     residual_value,
    #     adjusted_cap_cost,
    #     depreciation_fee,
    #     money_factor,
    # )

    # Option 2: LEASE TAX INCLUDED
    lt_net_cap_cost = residual_value + adjusted_cap_cost + total_taxes
    monthly_lt_included = (
        depreciation_fee + (lt_net_cap_cost * money_factor) + total_taxes / lease_term
    )
    option2_first = round(monthly_lt_included + fees_sum + bank_fee, 2)

    # Option 3: TAXES AND BANK
    tb_net_cap_cost = residual_value + adjusted_cap_cost + total_taxes + bank_fee
    monthly_taxes_bank = (
        depreciation_fee
        + (tb_net_cap_cost * money_factor)
        + (total_taxes + bank_fee) / lease_term
    )
    option3_first = round(monthly_taxes_bank + (dmv_fee + doc_fee), 2)

    # Option 4: First Due
    fd_net_cap_cost = (
        residual_value + adjusted_cap_cost + total_taxes + bank_fee + fees_sum
    )
    monthly_first_due = (
        depreciation_fee
        + (fd_net_cap_cost * money_factor)
        + (total_taxes + bank_fee + fees_sum) / lease_term
    )
    option4_first = round(monthly_first_due, 2)

    # Option 5: SIGN AND DRIVE $0 DAS
    if lease_term > 1:
        sad_net_cap_cost = (
            residual_value
            + adjusted_cap_cost
            + total_taxes
            + bank_fee
            + fees_sum
            + monthly_first_due
        )
        monthly_sign_drive = (
            depreciation_fee
            + (sad_net_cap_cost * money_factor)
            + (total_taxes + bank_fee + fees_sum + monthly_first_due) / (lease_term - 1)
        )
    else:
        monthly_sign_drive = 0
    option5_first = 0

    # Option 6: ONEPAY
    one_pay_monthly = depreciation_fee + (residual_value + adjusted_cap_cost) * (
        money_factor - 0.0008
    )
    onepay_total = (
        one_pay_monthly * lease_term
        + one_pay_monthly * tax_rate * lease_term
        + bank_fee
        + fees_sum
    )

    if lease_type == "TAXES AND FEES UPFRONT":
        return pd.Series([round(base_monthly, 2), option1_first])

    elif lease_type == "LEASE TAX INCLUDED":
        return pd.Series([round(monthly_lt_included, 2), option2_first])

    elif lease_type == "TAXES AND BANK":
        return pd.Series([round(monthly_taxes_bank, 2), option3_first])

    elif lease_type == "First Due":
        return pd.Series([round(monthly_first_due, 2), option4_first])

    elif lease_type == "SIGN AND DRIVE $0 DAS":
        return pd.Series([round(monthly_sign_drive, 2), 0])

    elif lease_type == "ONEPAY":
        return pd.Series([round(0, 2), onepay_total])
    else:
        return pd.Series([round(base_monthly, 2), option1_first])


filtered_data[["Monthly Payment", "Due at Signing"]] = filtered_data.apply(
    compute_lease, axis=1
)

if selected_makes:
    filtered_data = filtered_data[filtered_data["Make"].isin(selected_makes)]
if selected_years:
    filtered_data = filtered_data[filtered_data["Year"].isin(selected_years)]
if selected_models:
    filtered_data = filtered_data[filtered_data["Model"].isin(selected_models)]
if selected_trims:
    filtered_data = filtered_data[filtered_data["Trim"].isin(selected_trims)]
filtered_data = filtered_data[
    (filtered_data["MSRP"] >= selected_price_range[0])
    & (filtered_data["MSRP"] <= selected_price_range[1])
]

# -----------------------------
# Main Dashboard Content
# -----------------------------
st.title("Montage Auto Leasing Dashboard")

# --- 1. Available Cars ---
st.markdown("## 1. Available Cars")
if filtered_data.empty:
    st.warning("No cars found with the selected filters.")
else:
    # Display a subset of columns (adjust as needed)
    filtered_data[f"residual_value_{lease_term}"] = (
        filtered_data["MSRP"] * filtered_data[f"RP {lease_term}"] * 0.01
    )
    filtered_data["Year"] = filtered_data["Year"].apply(lambda x: f"{x}")

    display_cols = [
        col
        for col in [
            "Year",
            "Make",
            "Model",
            "Trim",
            "Rate Style",
            "Bank",
            "MSRP",
            "Monthly Payment",
            "First Payment",
            "Total_Rebates",
            "Package",
            "Rate Model",
            "Adjusted Cap Cost",
            f"residual_value_{lease_term}",
        ]
        if col in filtered_data.columns
    ]
    st.dataframe(filtered_data[display_cols])

# --- 2. Lease Computation ---
if not filtered_data.empty:
    st.markdown("## 2. Lease Computation")
    st.write("Select a car configuration for lease computation:")
    # Create selection options from the filtered data indices
    config_options = filtered_data.index.tolist()

    def format_option(idx):
        row = filtered_data.loc[idx]
        return f"{row['Make']} {row['Model']} {row['Trim']} — MSRP: {row['MSRP']}"

    selected_idx = st.selectbox(
        "Select a car configuration", options=config_options, format_func=format_option
    )
    selected_config = filtered_data.loc[selected_idx]

    # --- Lease Computation ---
    # The lease computation requires columns for residual percentage and money factor.
    # For a given lease term, these are assumed to be named like "RP 36" and "MF 36".
    rp_column = f"RP {lease_term}"
    mf_column = f"MF {lease_term}"

    if rp_column not in selected_config or mf_column not in selected_config:
        st.error(
            f"Required columns '{rp_column}' and/or '{mf_column}' not found in the data."
        )
    else:
        # Extract values from the selected configuration
        msrp = selected_config["MSRP"]
        adjusted_cap_cost = selected_config["Adjusted Cap Cost"]
        residual_percentage = selected_config[rp_column]  # e.g., 50 for 50%
        money_factor = selected_config[mf_column]
        residual_value = msrp * residual_percentage * 0.01

        # Fees and tax rate (adjust these as needed)
        # dmv_fee = 350
        # doc_fee = 249
        # bank_fee = 595
        # tax_rate = 0.08875
        fees_sum = dmv_fee + doc_fee

        depreciation_fee = (adjusted_cap_cost - residual_value) / lease_term
        # Option 1: TAXES AND FEES UPFRONT
        tfu_net_cap_cost = residual_value + adjusted_cap_cost
        base_monthly = depreciation_fee + (tfu_net_cap_cost * money_factor)
        total_taxes = base_monthly * tax_rate * lease_term
        option1_first = round(base_monthly + total_taxes + fees_sum + bank_fee, 2)
        # st.write(
        #     tfu_net_cap_cost,
        #     residual_value,
        #     adjusted_cap_cost,
        #     depreciation_fee,
        #     money_factor,
        # )

        # Option 2: LEASE TAX INCLUDED
        lt_net_cap_cost = residual_value + adjusted_cap_cost + total_taxes
        monthly_lt_included = (
            depreciation_fee
            + (lt_net_cap_cost * money_factor)
            + total_taxes / lease_term
        )
        option2_first = round(monthly_lt_included + fees_sum + bank_fee, 2)

        # Option 3: TAXES AND BANK
        tb_net_cap_cost = residual_value + adjusted_cap_cost + total_taxes + bank_fee
        monthly_taxes_bank = (
            depreciation_fee
            + (tb_net_cap_cost * money_factor)
            + (total_taxes + bank_fee) / lease_term
        )
        option3_first = round(monthly_taxes_bank + (dmv_fee + doc_fee), 2)

        # Option 4: First Due
        fd_net_cap_cost = (
            residual_value + adjusted_cap_cost + total_taxes + bank_fee + fees_sum
        )
        monthly_first_due = (
            depreciation_fee
            + (fd_net_cap_cost * money_factor)
            + (total_taxes + bank_fee + fees_sum) / lease_term
        )
        option4_first = round(monthly_first_due, 2)

        # Option 5: SIGN AND DRIVE $0 DAS
        if lease_term > 1:
            sad_net_cap_cost = (
                residual_value
                + adjusted_cap_cost
                + total_taxes
                + bank_fee
                + fees_sum
                + monthly_first_due
            )
            monthly_sign_drive = (
                depreciation_fee
                + (sad_net_cap_cost * money_factor)
                + (total_taxes + bank_fee + fees_sum + monthly_first_due)
                / (lease_term - 1)
            )
        else:
            monthly_sign_drive = 0
        option5_first = 0

        # Option 6: ONEPAY
        one_pay_monthly = depreciation_fee + (residual_value + adjusted_cap_cost) * (
            money_factor - 0.0008
        )
        onepay_total = (
            one_pay_monthly * lease_term
            + one_pay_monthly * tax_rate * lease_term
            + bank_fee
            + fees_sum
        )

        # Option 7: CUSTOM option
        st.markdown("### Custom Lease Option")
        input_type = st.radio(
            "Choose input method for custom lease option:",
            ["Set Monthly Payment", "Set First Payment"],
            key="custom_input",
        )
        if input_type == "Set Monthly Payment":
            custom_monthly = st.number_input(
                "Enter custom monthly payment",
                min_value=0.0,
                value=round(base_monthly, 2),
                key="custom_monthly",
            )

            custom_first = (
                -(
                    (
                        (lease_term * custom_monthly)
                        + ((1 - (lease_term * money_factor)) * residual_value)
                    )
                    / (1 + lease_term * money_factor)
                )
                + adjusted_cap_cost
                + custom_monthly
                + total_taxes
                + bank_fee
                + dmv_fee
                + doc_fee
            )
            custom_first = round(custom_first, 2)  # Auto-compute first payment

        else:
            custom_first = st.number_input(
                "Enter custom first payment",
                min_value=0.0,
                value=round(base_monthly + total_taxes + fees_sum, 2),
                key="custom_first",
            )
            custom_net_cap_cost = (
                adjusted_cap_cost
                + residual_value
                + total_taxes
                + bank_fee
                + fees_sum
                - custom_first
            )
            custom_monthly = (
                depreciation_fee
                + custom_net_cap_cost * money_factor
                + (custom_net_cap_cost - adjusted_cap_cost) / lease_term
            )
        # Assemble lease options into a dictionary
        lease_options = {
            "TAXES AND FEES UPFRONT": {
                "Monthly Payment": round(base_monthly, 2),
                "Bank Fee": bank_fee,
                "DMV Fee": dmv_fee,
                "Doc Fee": doc_fee,
                "Taxes": round(total_taxes, 2),
                "First Payment": option1_first,
            },
            "LEASE TAX INCLUDED": {
                "Monthly Payment": round(monthly_lt_included, 2),
                "Bank Fee": bank_fee,
                "DMV Fee": dmv_fee,
                "Doc Fee": doc_fee,
                "Taxes": 0,
                "First Payment": option2_first,
            },
            "TAXES AND BANK": {
                "Monthly Payment": round(monthly_taxes_bank, 2),
                "Bank Fee": 0,
                "DMV Fee": dmv_fee,
                "Doc Fee": doc_fee,
                "Taxes": 0,
                "First Payment": option3_first,
            },
            "First Due": {
                "Monthly Payment": round(monthly_first_due, 2),
                "Bank Fee": 0,
                "DMV Fee": 0,
                "Doc Fee": 0,
                "Taxes": 0,
                "First Payment": option4_first,
            },
            "SIGN AND DRIVE $0 DAS": {
                "Monthly Payment": round(monthly_sign_drive, 2),
                "Bank Fee": 0,
                "DMV Fee": 0,
                "Doc Fee": 0,
                "Taxes": 0,
                "First Payment": option5_first,
            },
            "ONEPAY": {
                "Monthly Payment": 0,
                "Bank Fee": bank_fee,
                "DMV Fee": dmv_fee,
                "Doc Fee": doc_fee,
                "Taxes": round(total_taxes, 2),
                "First Payment": round(onepay_total, 2),
            },
            "CUSTOM": {
                "Monthly Payment": custom_monthly,
                "Bank Fee": bank_fee,
                "DMV Fee": dmv_fee,
                "Doc Fee": doc_fee,
                "Taxes": round(total_taxes, 2),
                "First Payment": custom_first,
            },
        }

        # Convert the lease options to a DataFrame for display
        lease_df = pd.DataFrame(lease_options)
        lease_df = lease_df.reindex(
            [
                "Monthly Payment",
                "Bank Fee",
                "DMV Fee",
                "Doc Fee",
                "Taxes",
                "First Payment",
            ]
        )
        st.table(lease_df)


# Function to fetch API data (cached)
@st.cache_data
def fetch_car_data(api_key, selected_years, selected_makes, selected_models):
    BASE_URL = f"https://mc-api.marketcheck.com/v2/search/car/active?api_key={api_key}"

    params = {}
    if selected_years:
        params["year"] = ",".join(str(year) for year in selected_years)
    if selected_makes:
        params["make"] = ",".join(selected_makes)
    if selected_models:
        params["model"] = ",".join(selected_models)

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json().get("listings", [])
        return pd.DataFrame(data) if data else None
    else:
        st.error(f"Error fetching data from API. Status Code: {response.status_code}")
        return None


# --- Suggested Cars Section ---
st.markdown("## 3. Suggested Cars (from API)")

if st.button("Fetch Suggested Cars"):
    if api_key:
        with st.spinner("Fetching suggested cars..."):
            api_df = fetch_car_data(
                api_key, selected_years, selected_makes, selected_models
            )

            if api_df is not None:
                # Expand "build" column if present
                if "build" in api_df.columns:
                    build_expanded = api_df["build"].apply(pd.Series)
                    api_df = pd.concat(
                        [api_df.drop(columns=["build"]), build_expanded], axis=1
                    )

                # Sort by MSRP
                if "msrp" in api_df.columns:
                    api_df.sort_values(by="msrp", inplace=True, na_position="last")

                # Store data in session state
                st.session_state["api_df"] = api_df
            else:
                st.session_state["api_df"] = None
                st.info("No suggested cars found from API with the given parameters.")

# Retain fetched data in session state
if "api_df" in st.session_state and st.session_state["api_df"] is not None:
    api_df = st.session_state["api_df"]

    # Sidebar filters (do not trigger re-fetch)
    if "interior_color" in api_df.columns:
        int_color_options = sorted(api_df["interior_color"].dropna().unique())
        selected_int_colors = st.sidebar.multiselect(
            "Select Interior Color", options=int_color_options, default=[]
        )

    if "exterior_color" in api_df.columns:
        ext_color_options = sorted(api_df["exterior_color"].dropna().unique())
        selected_ext_colors = st.sidebar.multiselect(
            "Select Exterior Color", options=ext_color_options, default=[]
        )

    # Apply filters dynamically
    if selected_int_colors:
        api_df = api_df[api_df["interior_color"].isin(selected_int_colors)]
    if selected_ext_colors:
        api_df = api_df[api_df["exterior_color"].isin(selected_ext_colors)]

    first_photo_link = api_df["media"].apply(
        lambda x: (
            x[0]["photo_links"][0]
            if isinstance(x, list) and x and "photo_links" in x[0]
            else None
        )
    )

    # Define display columns
    display_columns = [
        "year",
        "make",
        "model",
        "trim",
        "version",
        "body_type",
        "vehicle_type",
        "dom",
        "vin",
        "transmission",
        "drivetrain",
        "fuel_type",
        "engine",
        "heading",
        "msrp",
        "price",
        "miles",
        "exterior_color",
        "interior_color",
    ]

    # Add extracted media column
    api_df["first_media_photo"] = first_photo_link
    display_columns.append("first_media_photo")

    # Ensure columns exist in DataFrame
    existing_columns = [col for col in display_columns if col in api_df.columns]

    # Display DataFrame
    st.dataframe(api_df[existing_columns])

# -----------------------------
# Footer / Contact Information
# -----------------------------
st.markdown("---")
st.markdown(
    "### Contact us for more information at: [montage.auto@email.com](mailto:montage.auto@email.com)"
)
