import pandas as pd

print("Loading SOCATv2025 data...")
df = pd.read_csv(
    '/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/SOCATv2025_SO.tsv',
    skiprows=1499,
    sep='\t'
)
print(f"Original dataframe loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# Columns to keep
columns_to_keep = [
    'Expocode', 'yr', 'mon', 'day', 'hh', 'mm', 'ss',
    'longitude [dec.deg.E]', 'latitude [dec.deg.N]',
    'fCO2rec [uatm]', 'sample_depth [m]', 'sal', 'SST [deg.C]', 'PPPP [hPa]'
]

print("Filtering and renaming columns...")
df = (
    df[columns_to_keep]
    .rename(columns={
        'Expocode': 'expocode',
        'yr': 'year',
        'mon': 'month',
        'day': 'day',
        'hh': 'hour',
        'mm': 'minute',
        'ss': 'second',
        'longitude [dec.deg.E]': 'longitude',
        'latitude [dec.deg.N]': 'latitude',
        'fCO2rec [uatm]': 'fco2_rec',
        'sample_depth [m]': 'sample_depth_m',
        'sal': 'salinity',
        'SST [deg.C]': 'sst_degC',
        'PPPP [hPa]': 'pressure_hpa'
    })
    .dropna(subset=['salinity', 'sst_degC', 'fco2_rec'])
)

print(f"Filtered dataframe has {df.shape[0]} rows after dropping missing values.")

# Save as CSV
output_path = '/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/SOCATv2025_SO_clean.csv'
df.to_csv(output_path, index=False)
print(f"Cleaned data saved to: {output_path}")
