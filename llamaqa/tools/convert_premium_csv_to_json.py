import json

import pandas as pd


def parse_header(header):
    """Parse the column header to extract company and plan."""
    parts = header.split(" - ")
    if len(parts) == 2:
        company, plan = parts
        return company, plan
    return None, None


def csv_to_json(csv_path, json_file_path):
    # Initialize the structure for the JSON data
    structured_data = {}

    # Process data
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        age = str(row["Age Next Birthday"])  # Age as a string to match the JSON format
        if age == ">100":
            age = "Over 100"

        # Initialize the age-specific data
        if age not in structured_data:
            structured_data[age] = {
                "medishield_premium": float(
                    row["MediShield Life Premiums (Fully payable by Medisave)"]
                ),
                "additional_withdrawal_limit": float(
                    row["Additional Withdrawal Limits (AWLs)"]
                ),
                "companies": {},
            }

        # Process each company-plan column in the data row
        for col in df.columns[3:]:  # Skip 'Age', 'MediShield Life Premiums', and 'AWLs'
            company, plan = parse_header(col)
            if company and plan:
                if company not in structured_data[age]["companies"]:
                    structured_data[age]["companies"][company] = {}
                if plan not in structured_data[age]["companies"][company]:
                    structured_data[age]["companies"][company][plan] = {}

                # Convert the value to a numeric type, coercing errors to NaN
                value = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(value):  # If the value is a number
                    structured_data[age]["companies"][company][plan][
                        "additional_insurance_premium"
                    ] = float(value)
                else:  # If the value is not a number, keep it as a string
                    structured_data[age]["companies"][company][plan][
                        "additional_insurance_premium"
                    ] = str(row[col])

    # Write the structured data to a JSON file
    with open(json_file_path, "w") as jsonfile:
        json.dump(structured_data, jsonfile, indent=4)


# Example usage
csv_to_json("premiums.csv", "premiums.json")
