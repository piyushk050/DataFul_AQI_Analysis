import os
import pandas as pd
import numpy as np

# Initialize explicit output destination path
output_dir = "clean_dataset/dashboard_outputs"
os.makedirs(output_dir, exist_ok=True)

print("--- Step 1: Loading Cleaned Data Layers ---")
# Accessing target data fields from your local clean folder architecture
df_aqi = pd.read_csv("clean_dataset/cleaned_aqi_data.csv")
df_disease = pd.read_csv("clean_dataset/cleaned_health_data.csv")
df_vehicles = pd.read_csv("clean_dataset/cleaned_vehicle_data.csv")
df_pop = pd.read_csv("clean_dataset/cleaned_population_data.csv")

# Standardize date columns safely into datetime objects
df_aqi['clean_date'] = pd.to_datetime(df_aqi['date'], errors='coerce')
df_disease['clean_start_date'] = pd.to_datetime(df_disease['outbreak_starting_date'], errors='coerce')

print("Datasets mapped and indexed successfully.\n")

# =====================================================================
# Q1: TOP 5 AND BOTTOM 5 AREAS BY AVERAGE AQI (Dec 2024 - May 2025)
# =====================================================================
print("--- Computing Question 1: AQI Severity Mapping ---")
q1_window = df_aqi[(df_aqi['clean_date'] >= '2024-12-01') & (df_aqi['clean_date'] <= '2025-05-31')]
q1_grouped = q1_window.groupby(['area', 'state'])['aqi_value'].mean().reset_index()

top_5_aqi = q1_grouped.sort_values(by='aqi_value', ascending=False).head(5)
bottom_5_aqi = q1_grouped.sort_values(by='aqi_value', ascending=True).head(5)

top_5_aqi.to_csv(f"{output_dir}/q1_top_5_polluted.csv", index=False)
bottom_5_aqi.to_csv(f"{output_dir}/q1_bottom_5_cleanest.csv", index=False)
print("-> Q1 Complete: Severity target slices stored safely.")

# =====================================================================
# Q2: TOP 2 AND BOTTOM 2 PROMINENT POLLUTANTS FOR SOUTHERN INDIA
# =====================================================================
print("\n--- Computing Question 2: Pollutant Distribution for Southern India ---")
south_states = ['Karnataka', 'Tamil Nadu', 'Kerala', 'Andhra Pradesh', 'Telangana']
q2_window = df_aqi[(df_aqi['state'].isin(south_states)) & (df_aqi['clean_date'].dt.year >= 2022)].copy()
q2_window = q2_window[q2_window['prominent_pollutants'].notna() & (q2_window['prominent_pollutants'] != 'None')]

pollutant_counts = q2_window.groupby(['state', 'prominent_pollutants']).size().reset_index(name='days_tracked')
pollutant_counts['rank_desc'] = pollutant_counts.groupby('state')['days_tracked'].rank(method='dense', ascending=False)
pollutant_counts['rank_asc'] = pollutant_counts.groupby('state')['days_tracked'].rank(method='dense', ascending=True)

top_2_pollutants = pollutant_counts[pollutant_counts['rank_desc'] <= 2]
bottom_2_pollutants = pollutant_counts[pollutant_counts['rank_asc'] <= 2]

top_2_pollutants.to_csv(f"{output_dir}/q2_top_pollutants_south.csv", index=False)
bottom_2_pollutants.to_csv(f"{output_dir}/q2_bottom_pollutants_south.csv", index=False)
print("-> Q2 Complete: Target particle structures mapped.")

# =====================================================================
# Q3: WEEKEND VS WEEKDAY AQI IN METRO CITIES (SettingWithCopy Fixed)
# =====================================================================
print("\n--- Computing Question 3: Weekend vs. Weekday Traffic Footprints ---")
metro_hubs = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Pune']
max_date = df_aqi['clean_date'].max()

# Use explicit .copy() to neutralize the SettingWithCopy slicing warning flag
q3_window = df_aqi[(df_aqi['area'].isin(metro_hubs)) & (df_aqi['clean_date'] >= (max_date - pd.DateOffset(years=1)))].copy()

# Element calculation maps directly to standard part-of-week labels
q3_window['day_type'] = q3_window['clean_date'].dt.dayofweek.isin([5, 6]).map({True: 'Weekend', False: 'Weekday'})
q3_grouped = q3_window.groupby(['area', 'day_type'])['aqi_value'].mean().unstack().reset_index()

q3_grouped.to_csv(f"{output_dir}/q3_metro_weekend_analysis.csv", index=False)
print("-> Q3 Complete: Weekly commuter divergence metrics evaluated.")

# =====================================================================
# Q4: MONTHS WITH THE WORST AIR QUALITY ACROSS TOP 10 STATES
# =====================================================================
print("\n--- Computing Question 4: Seasonal Air Quality Baselines ---")
top_10_states = df_aqi.groupby('state')['area'].nunique().nlargest(10).index
q4_window = df_aqi[df_aqi['state'].isin(top_10_states)].copy()

q4_grouped = q4_window.groupby(['state', q4_window['clean_date'].dt.month])['aqi_value'].mean().reset_index()
q4_grouped.rename(columns={'clean_date': 'month_num'}, inplace=True)
q4_grouped.to_csv(f"{output_dir}/q4_seasonal_trends.csv", index=False)
print("-> Q4 Complete: Regional atmospheric variance matrix computed.")

# =====================================================================
# Q5: BENGALURU AIR QUALITY CATEGORY DISTRIBUTION
# =====================================================================
print("\n--- Computing Question 5: Categorical Status Counts for Bengaluru ---")
q5_window = df_aqi[(df_aqi['area'] == 'Bengaluru') & (df_aqi['clean_date'] >= '2025-03-01') & (df_aqi['clean_date'] <= '2025-05-31')]
q5_summary = q5_window['air_quality_status'].value_counts().reset_index()
q5_summary.columns = ['air_quality_status', 'days_count']

q5_summary.to_csv(f"{output_dir}/q5_bengaluru_status_counts.csv", index=False)
print("-> Q5 Complete: Localized urban status intervals exported.")

# =====================================================================
# Q6: TOP 2 DISEASE OUTBREAKS VS REGIONAL HISTORICAL AQI
# =====================================================================
print("\n--- Computing Question 6: Relational Public Health Outbreak Link ---")
df_disease_recent = df_disease[df_disease['year'] >= (df_disease['year'].max() - 2)]
disease_grouped = df_disease_recent.groupby(['state', 'disease_illness_name'])['cases'].sum().reset_index()

disease_grouped['rnk'] = disease_grouped.groupby('state')['cases'].rank(method='dense', ascending=False)
top_2_diseases = disease_grouped[disease_grouped['rnk'] <= 2]

state_aqi_baseline = df_aqi.groupby('state')['aqi_value'].mean().reset_index().rename(columns={'aqi_value': 'historical_aqi_mean'})
q6_master_df = pd.merge(top_2_diseases, state_aqi_baseline, on='state', how='inner')

q6_master_df.to_csv(f"{output_dir}/q6_epidemiology_environmental_link.csv", index=False)
print("-> Q6 Complete: Epidemiological impact baseline constructed.")

# =====================================================================
# Q7: EV ADOPTION VS AQI (Unique Bins ValueError Fixed)
# =====================================================================
print("\n--- Computing Question 7: Clean-Energy EV Adoption Tracker ---")
# Standardize fuel strings to safely avoid value mismatching
df_vehicles['fuel_clean'] = df_vehicles['fuel'].astype(str).str.upper().str.strip()

# Flexible lookup query string handles variations like 'Electric' or 'Pure EV'
ev_logic = df_vehicles['fuel_clean'].str.contains('ELECT|EV|HYBRID', na=False)
df_ev_totals = df_vehicles[ev_logic].groupby('state')['value'].sum().reset_index()
df_ev_totals.rename(columns={'value': 'total_ev_registrations'}, inplace=True)

# Relational framework merge execution
q7_master_df = pd.merge(df_ev_totals, state_aqi_baseline, on='state', how='inner')

# Clean out potential missing records
q7_master_df = q7_master_df.dropna(subset=['total_ev_registrations'])

# Fallback error engine routes bin segmentation safely if distinct counts exist
if q7_master_df['total_ev_registrations'].nunique() > 1:
    q7_master_df['adoption_tier'] = pd.qcut(
        q7_master_df['total_ev_registrations'], 
        q=2, 
        labels=['Lower EV Adoption', 'High EV Adoption'],
        duplicates='drop'
    )
else:
    q7_master_df['adoption_tier'] = 'Baseline Adoption'

q7_master_df.to_csv(f"{output_dir}/q7_ev_infrastructure_performance.csv", index=False)
print("-> Q7 Complete: Clean energy infrastructure analysis executed.")

print("\n=====================================================================")
print("🎉 SUCCESS: THE COMPLETE PRIMARY DATA ENGINE HAS CONCLUDED ERROR-FREE!")
print(f"All pre-aggregated CSV logs are saved inside: '{output_dir}/'")
print("=====================================================================")
