import os
import pandas as pd
import numpy as np

# Automatically detects the exact folder where this script is saved!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_existing_path(base_name):
    """Checks for both single and double .csv extensions dynamically."""
    path_single = os.path.join(BASE_DIR, f"{base_name}.csv")
    path_double = os.path.join(BASE_DIR, f"{base_name}.csv.csv")
    
    if os.path.exists(path_single):
        return path_single
    elif os.path.exists(path_double):
        return path_double
    else:
        return path_single

def smart_read_csv(file_path):
    """Reads CSV files handling multiple encoding types safely."""
    encodings = ['utf-8', 'cp1252', 'latin-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # If all encodings fail, raise the original error
    return pd.read_csv(file_path)

# Find paths dynamically resolving double extension errors
AQI_FILE = get_existing_path("source_1")
HEALTH_FILE = get_existing_path("source_2")
VEHICLE_FILE = get_existing_path("source_3")
POPULATION_FILE = get_existing_path("source_4")

# ----------------------------------------------------
# 1. CLEANING FILE 1: AQI DATA
# ----------------------------------------------------
def clean_aqi_data(file_path):
    print(f"Opening AQI Data from: {os.path.basename(file_path)}")
    df = smart_read_csv(file_path)
    
    # Standardize Dates to YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
    
    # Convert numerical values to integers
    df['number_of_monitoring_stations'] = df['number_of_monitoring_stations'].fillna(0).astype(int)
    df['aqi_value'] = df['aqi_value'].fillna(0).astype(int)
    
    # Strip string white spaces
    df['state'] = df['state'].str.strip()
    df['area'] = df['area'].str.strip()
    df['air_quality_status'] = df['air_quality_status'].str.strip()
    
    # Drop redundant columns
    df = df.drop(columns=['unit', 'note'], errors='ignore')
    
    output_path = os.path.join(BASE_DIR, "cleaned_aqi_data.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved -> {output_path}\n")

# ----------------------------------------------------
# 2. CLEANING FILE 2: HEALTH OUTBREAKS DATA
# ----------------------------------------------------
def clean_health_data(file_path):
    print(f"Opening Health Data from: {os.path.basename(file_path)}")
    df = smart_read_csv(file_path)
    
    # Standardize Dates
    df['outbreak_starting_date'] = pd.to_datetime(df['outbreak_starting_date'], format='%d-%m-%Y', errors='coerce').dt.strftime('%Y-%m-%d')
    df['reporting_date'] = pd.to_datetime(df['reporting_date'], format='%d-%m-%Y', errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Fill missing reporting dates with starting date as a logical proxy
    df['reporting_date'] = df['reporting_date'].fillna(df['outbreak_starting_date'])
    
    # Standardize state and district name typos
    df['state'] = df['state'].str.strip().replace({'Madhya': 'Madhya Pradesh'})
    df['district'] = df['district'].str.strip().replace({
        'East Singhbum': 'East Singhbhum',
        'Purba Barshaman': 'Purba Bardhaman'
    })
    
    # Fix corrupted disease name entries
    df['disease_illness_name'] = df['disease_illness_name'].str.strip().replace({'Kannad': 'Kyasanur Forest Disease'})
    
    # Drop redundant tracking columns
    df = df.drop(columns=['unit', 'note'], errors='ignore')
    
    output_path = os.path.join(BASE_DIR, "cleaned_health_data.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved -> {output_path}\n")

# ----------------------------------------------------
# 3. CLEANING FILE 3: VEHICLE ADOPTION DATA
# ----------------------------------------------------
def clean_vehicle_data(file_path):
    print(f"Opening Vehicle Data from: {os.path.basename(file_path)}")
    df = smart_read_csv(file_path)
    
    # Standardize fuel classification categories
    df['fuel'] = df['fuel'].str.strip().replace({
        'ELECTRIC(BOV)': 'EV',
        'PURE EV': 'EV',
        'STRONG HYBRID EV': 'Hybrid',
        'PETROL/HYBRID': 'Hybrid',
        'DIESEL/HYBRID': 'Hybrid',
        'PETROL/ETHANOL': 'Ethanol Blend'
    })
    
    # Clean text columns
    df['state'] = df['state'].str.strip()
    df['vehicle_class'] = df['vehicle_class'].str.strip()
    
    # Drop completely static RTO registry descriptions and notes
    df = df.drop(columns=['rto', 'unit', 'note'], errors='ignore')
    
    output_path = os.path.join(BASE_DIR, "cleaned_vehicle_data.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved -> {output_path}\n")

# ----------------------------------------------------
# 4. CLEANING FILE 4: POPULATION PROJECTION DATA
# ----------------------------------------------------
def clean_population_data(file_path):
    print(f"Opening Population Data from: {os.path.basename(file_path)}")
    df = smart_read_csv(file_path)
    
    # Convert scaled projection figures back to true absolute metrics
    df['value'] = df['value'].astype(int) * 1000
    
    # Clean textual fields
    df['state'] = df['state'].str.strip()
    df['month'] = df['month'].str.strip()
    
    # Drop units notation column
    df = df.drop(columns=['unit', 'note'], errors='ignore')
    
    output_path = os.path.join(BASE_DIR, "cleaned_population_data.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved -> {output_path}\n")

# Run all workflows sequentially
if __name__ == "__main__":
    try:
        clean_aqi_data(AQI_FILE)
        clean_health_data(HEALTH_FILE)
        clean_vehicle_data(VEHICLE_FILE)
        clean_population_data(POPULATION_FILE)
        print("🎉 Success! All data files cleaned and optimized for your project.")
    except FileNotFoundError as e:
        print(f"\n❌ Execution Error: {e}")
        print("Please verify the names of your files inside your directory.")
