import pandas as pd


def parse_float(value, default):
    """Convert input to float, return default if invalid."""
    try:
        return float(value)
    except ValueError:
        return default


def compute_lease(row, lease_term, tax_rate, dmv_fee, doc_fee, bank_fee, lease_type):
    """Calculate lease payments based on the selected lease term."""
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

    # Lease Tax Included
    lt_net_cap_cost = residual_value + adjusted_cap_cost + total_taxes
    monthly_lt_included = (
        depreciation_fee + (lt_net_cap_cost * money_factor) + total_taxes / lease_term
    )
    option2_first = round(monthly_lt_included + fees_sum + bank_fee, 2)

    # Taxes and Bank
    tb_net_cap_cost = residual_value + adjusted_cap_cost + total_taxes + bank_fee
    monthly_taxes_bank = (
        depreciation_fee
        + (tb_net_cap_cost * money_factor)
        + (total_taxes + bank_fee) / lease_term
    )
    option3_first = round(monthly_taxes_bank + (dmv_fee + doc_fee), 2)

    # First Due
    fd_net_cap_cost = (
        residual_value + adjusted_cap_cost + total_taxes + bank_fee + fees_sum
    )
    monthly_first_due = (
        depreciation_fee
        + (fd_net_cap_cost * money_factor)
        + (total_taxes + bank_fee + fees_sum) / lease_term
    )
    option4_first = round(monthly_first_due, 2)

    # SIGN AND DRIVE $0 DAS
    option5_first = 0
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

    # ONEPAY
    one_pay_monthly = depreciation_fee + (residual_value + adjusted_cap_cost) * (
        money_factor - 0.0008
    )
    onepay_total = (
        one_pay_monthly * lease_term
        + one_pay_monthly * tax_rate * lease_term
        + bank_fee
        + fees_sum
    )

    # Determine the return values based on lease type
    options = {
        "TAXES AND FEES UPFRONT": [round(base_monthly, 2), option1_first],
        "LEASE TAX INCLUDED": [round(monthly_lt_included, 2), option2_first],
        "TAXES AND BANK": [round(monthly_taxes_bank, 2), option3_first],
        "First Due": [round(monthly_first_due, 2), option4_first],
        "SIGN AND DRIVE $0 DAS": [round(monthly_sign_drive, 2), option5_first],
        "ONEPAY": [0, onepay_total],
    }

    return pd.Series(options.get(lease_type, [round(base_monthly, 2), option1_first]))
