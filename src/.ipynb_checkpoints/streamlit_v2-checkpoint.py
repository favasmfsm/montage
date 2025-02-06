import streamlit as st
import pandas as pd

# -----------------------------
# Main Streamlit App
# -----------------------------
# Set page configuration
st.set_page_config(page_title="Montage Auto Leasing", page_icon=":car:", layout="wide")

# Display a banner and logo
st.image('src/banner.png', use_column_width=True)
st.sidebar.image('src/logo.png', width=150)

# -----------------------------
# Data Loading and Filtering
# -----------------------------
# Load data


    
df = pd.read_csv("./data/merged_data_v2.csv")

# Sidebar selections (filtering)
st.sidebar.header("Filter Options")

# Make filter (optional)
make_options = ["All"] + list(df['Make'].unique())
make = st.sidebar.selectbox("Select Make", make_options)

# Apply Make filter if selected
filtered_data = df if make == "All" else df[df['Make'] == make]

# Year filter (optional)
year_options = ["All"] + sorted(filtered_data['Year'].unique().tolist())
year = st.sidebar.selectbox("Select Year", year_options)

# Apply Year filter if selected
if year != "All":
    filtered_data = filtered_data[filtered_data['Year'] == year]

# Model filter (optional)
model_options = ["All"] + sorted(filtered_data['Model'].unique().tolist())
model = st.sidebar.selectbox("Select Model", model_options)

# Apply Model filter if selected
if model != "All":
    filtered_data = filtered_data[filtered_data['Model'] == model]

# Trim filter (optional)
trim_options = ["All"] + sorted(filtered_data['Trim'].unique().tolist())
trim = st.sidebar.selectbox("Select Trim", trim_options)

# Apply Trim filter if selected
if trim != "All":
    filtered_data = filtered_data[filtered_data['Trim'] == trim]

# Lease Term filter
months = st.sidebar.selectbox("Select Lease Term in months", [24, 27, 30, 33, 36, 39, 42, 48])

st.markdown("## Available Cars in this Configuration")
st.table(filtered_data[["Package", "Rate Model", "Bank", "MSRP","Adjusted Cap Cost"]])

# -----------------------------
# Allow User to Pick a Specific Car Configuration
# -----------------------------
if not filtered_data.empty:
    # Create a list of options for the selectbox.
    # Here, we format each option to show key details.
    options = filtered_data.index.tolist()
    def format_option(idx):
        row = filtered_data.loc[idx]
        return f"Package: {row['Package']} | Rate Model: {row['Rate Model']} | Bank: {row['Bank']} | MSRP: {row['MSRP']}"
    
    selected_index = st.selectbox("Select a car configuration", options, format_func=format_option)
    selected_config = filtered_data.loc[selected_index]

    # -----------------------------
    # Constant Fees and Additional Fees
    # -----------------------------
    dmv_fee = 350
    doc_fee = 249
    bank_fee = 100         # Sample value
    xmc = 200              # Sample value
    disposition_fee = 150  # Sample value
    tax_rate = 0.08875

    # -----------------------------
    # Base Calculations using the Selected Car's Data
    # -----------------------------
    msrp = selected_config['MSRP']
    adjusted_cap_cost = selected_config['Adjusted Cap Cost']
    
    # Get the appropriate residual percentage and money factor column based on the term
    rp_column = f'RP {months}'
    mf_column = f'MF {months}'
    
    # Calculate the residual value
    residual_value = msrp * selected_config[rp_column] * 0.01

    money_factor = selected_config[mf_column]
    fees_sum = dmv_fee + doc_fee


    def adj_capcost_to_monthly(new_adjusted_cap_cost,months):
        

        depreciation_fee = (new_adjusted_cap_cost - residual_value) / months
        finance_fee = (new_adjusted_cap_cost + residual_value) * money_factor
        base_monthly = depreciation_fee + finance_fee

        return base_monthly

    

    # -----------------------------
    # Compute Lease Options
    # -----------------------------
    lease_options = {}

    # 1. TAXES AND FEES UPFRONT
    base_monthly = adj_capcost_to_monthly(adjusted_cap_cost,months)
    total_taxes = base_monthly * tax_rate * months
    lease_options["TAXES AND FEES UPFRONT"] = {
        "Monthly Payment": round(base_monthly, 2),
        "Bank Fee": bank_fee,
        "DMV Fee": dmv_fee,
        "Doc Fee": doc_fee,
        "Taxes":total_taxes,
        "First Payment": round(base_monthly + total_taxes + fees_sum + bank_fee, 2)
    }

    # 2. LEASE TAX INCLUDED
    monthly_lt_included = adj_capcost_to_monthly(adjusted_cap_cost+total_taxes,months)
    lease_options["LEASE TAX INCLUDED"] = {
        "Monthly Payment": round(monthly_lt_included, 2),
        "Bank Fee": bank_fee,
        "DMV Fee": dmv_fee,
        "Doc Fee": doc_fee,
        "Taxes":0,
        "First Payment": round(monthly_lt_included + fees_sum + bank_fee, 2)
    }

    # 3. TAXES AND BANK
    monthly_taxes_bank = adj_capcost_to_monthly(adjusted_cap_cost+total_taxes+bank_fee,months)
    lease_options["TAXES AND BANK"] = {
        "Monthly Payment": round(monthly_taxes_bank, 2),
        "Bank Fee": 0,  # financed
        "DMV Fee": dmv_fee,
        "Doc Fee": doc_fee,
        "Taxes":0,
        "First Payment": round(monthly_taxes_bank + (dmv_fee + doc_fee), 2)
    }

    # 4. First Due
    monthly_first_due = adj_capcost_to_monthly(adjusted_cap_cost+total_taxes+bank_fee+fees_sum,months)
    lease_options["First Due"] = {
        "Monthly Payment": round(monthly_first_due, 2),
        "Bank Fee": 0,
        "DMV Fee": 0,
        "Doc Fee": 0,
        "Taxes":0,
        "First Payment": round(monthly_first_due, 2)
    }

    # 5. SIGN AND DRIVE $0 DAS
    monthly_sign_drive = adj_capcost_to_monthly(adjusted_cap_cost+total_taxes+bank_fee+fees_sum+monthly_first_due,months-1)
    lease_options["SIGN AND DRIVE $0 DAS"] = {
        "Monthly Payment": round(monthly_sign_drive, 2),
        "Bank Fee": 0,
        "DMV Fee": 0,
        "Doc Fee": 0,
        "Taxes":0,
        "First Payment": 0
    }
    depreciation_fee = (adjusted_cap_cost - residual_value) / months
    finance_fee = (adjusted_cap_cost) * money_factor
    base_monthly_on_pay = depreciation_fee + finance_fee
    
    onepay_total = base_monthly*(months-1)+ total_taxes + bank_fee + fees_sum
    # 6. ONEPAY
    lease_options["ONEPAY"] = {
        "Monthly Payment": 0,
        "Bank Fee": bank_fee,
        "DMV Fee": dmv_fee,
        "Doc Fee": doc_fee,
        "Taxes":round(total_taxes,2),
        "First Payment": round(onepay_total, 2)
    }

    # 7. CUSTOM 
    # For the CUSTOM option, you could allow the user to input their own monthly payment and/or first payment.
    input_type = st.radio("Choose input method:", ["Set Monthly Payment", "Set First Payment"])
    
    if input_type == "Set Monthly Payment":
        custom_monthly = st.number_input("Enter custom monthly payment", min_value=0.0,max_value=monthly_sign_drive, value=round(base_monthly, 2))
        
        custom_first = -(((months*custom_monthly)+((1-(months*money_factor))*residual_value))/(1+months*money_factor))+adjusted_cap_cost+custom_monthly+total_taxes + bank_fee+dmv_fee+doc_fee
        custom_first = round(custom_first, 2)  # Auto-compute first payment
    else:
        custom_first = st.number_input("Enter custom first payment", min_value=0.0, value=round(base_monthly + total_taxes + fees_sum, 2))
        custom_monthly = round(adj_capcost_to_monthly(adjusted_cap_cost+total_taxes+bank_fee+fees_sum-custom_first,months), 2)  # Auto-compute monthly payment

    lease_options["CUSTOM"] = {
        "Monthly Payment": custom_monthly,
        "Bank Fee": bank_fee,
        "DMV Fee": dmv_fee,
        "Doc Fee": doc_fee,
        "Taxes":round(total_taxes,2),
        "First Payment": custom_first
    }

    # -----------------------------
    # Create a DataFrame for Display
    # -----------------------------
    table_data = pd.DataFrame(lease_options, index=[
        "Monthly Payment",
        "Bank Fee",
        "DMV Fee",
        "Doc Fee",
        "Taxes",
        "First Payment"
    ])

    st.markdown("## Sample Lease Computation Table")
    st.markdown("Below is a table showing various lease payment configurations:")
    st.table(table_data)

st.markdown("---")
st.markdown("### Contact us for more information at: [montage.auto@email.com](mailto:montage.auto@email.com)")
