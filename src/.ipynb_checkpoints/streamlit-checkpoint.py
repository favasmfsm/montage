import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv("./data/merged_data_v2.csv")

# Set page configuration for a more appealing layout
st.set_page_config(page_title="Montage Auto Leasing", page_icon=":car:", layout="wide")

# Display the banner at the top
st.image('src/banner.png', use_column_width=True)

# Display logo in the sidebar
st.sidebar.image('src/logo.png', width=150)

make = st.sidebar.selectbox("Select Make", df['Make'].unique())

# Filter Year options based on selected Make
year = st.sidebar.selectbox("Select Year", df[df['Make'] == make]['Year'].unique())

# Filter Model options based on selected Make and Year
model = st.sidebar.selectbox("Select Model", df[(df['Make'] == make) & (df['Year'] == year)]['Model'].unique())

# Filter Trim options based on selected Make, Year, and Model
trim = st.sidebar.selectbox("Select Trim", df[(df['Make'] == make) & (df['Year'] == year) & (df['Model'] == model)]['Trim'].unique())

# Lease term in months
months = st.sidebar.selectbox("Select Lease Term in months", [24, 27, 30, 33, 36, 39, 42, 48])

# Filter data progressively based on selections
filtered_data = df
filtered_data = filtered_data[filtered_data['Make'] == make]
filtered_data = filtered_data[filtered_data['Year'] == year]
filtered_data = filtered_data[filtered_data['Model'] == model]
filtered_data = filtered_data[filtered_data['Trim'] == trim]

# Check if the filtered data is available and compute values
if not filtered_data.empty:
    # Extract values from the selected row
    msrp = filtered_data['MSRP'].values[0]
    adjusted_cap_cost = filtered_data['Adjusted Cap Cost'].values[0]
    
    # Calculate residual value
    rp_column = f'RP {months}'
    residual_value = msrp * filtered_data[rp_column].values[0] * 0.01

    # Money factor calculation
    mf_column = f'MF {months}'
    money_factor = filtered_data[mf_column].values[0]

    # Depreciation fee (Lease cost)
    depreciation_fee = (adjusted_cap_cost - residual_value) / months

    # Finance fee (Rent charge)
    finance_fee = (adjusted_cap_cost + residual_value) * money_factor

    # Tax rate and taxes calculation
    tax_rate = 8.875 / 100
    taxes = (depreciation_fee + finance_fee) * tax_rate

    # Total monthly lease payment
    # monthly_payment = depreciation_fee + finance_fee + taxes
    monthly_payment = depreciation_fee + finance_fee

    # Display the computed results in a clear, attractive layout
    st.markdown(f"### **Selected Package:** {trim} {model} {make} {year}")
    st.markdown(f"**MSRP:** ${msrp:,.2f}")
    st.markdown(f"**Adjusted Cap Cost:** ${adjusted_cap_cost:,.2f}")
    st.markdown(f"**Residual Package (RP {months}):** {filtered_data[rp_column].values[0]:,.2f}%")
    st.markdown(f"**Residual Value (RP {months}):** ${residual_value:,.2f}")
    st.markdown(f"**Money Factor (MF {months}):** {money_factor}")
    st.markdown(f"**Depreciation Fee:** ${depreciation_fee:,.2f}")
    st.markdown(f"**Finance Fee:** ${finance_fee:,.2f}")
    # st.markdown(f"**Taxes (8.875%):** ${taxes:,.2f}")
    st.markdown(f"### **Total Monthly Lease Payment:** ${monthly_payment:,.2f}")
else:
    st.write("No data found for the selected options.")

# Add a footer with additional contact or business info
st.markdown("---")
st.markdown("### Contact us for more information at: [montage.auto@email.com](mailto:montage.auto@email.com)")

