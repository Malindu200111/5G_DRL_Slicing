import pandas as pd

print("1. Loading selected CSV files from data/ folder...")
try:
    df_video = pd.read_csv('data/video_traffic.csv')
    df_gaming = pd.read_csv('data/gaming_traffic.csv')
    df_iot = pd.read_csv('data/iot_traffic.csv')
    print("Files loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

def extract_seconds(df):
    time_col = None
    for col in df.columns:
        if col.lower() in ['timestamp', 'time', 'frame', 'sec', 'second']:
            time_col = col
            break
    if not time_col:
        time_col = df.columns[0]

    # Check if numeric or datetime string
    series = df[time_col]
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    else:
        # Convert datetime strings to seconds relative to start
        dt_series = pd.to_datetime(series, errors='coerce')
        start_time = dt_series.min()
        seconds_series = (dt_series - start_time).dt.total_seconds().fillna(0).astype(int)
        return seconds_series

print(r"2. Calculating 1-second arrival rates (\lambda)...")

video_sec = extract_seconds(df_video)
gaming_sec = extract_seconds(df_gaming)
iot_sec = extract_seconds(df_iot)

embb_lambda = df_video.groupby(video_sec).size().reset_index(name='embb_lambda')
embb_lambda.columns = ['sec', 'embb_lambda']

urllc_lambda = df_gaming.groupby(gaming_sec).size().reset_index(name='urllc_lambda')
urllc_lambda.columns = ['sec', 'urllc_lambda']

if 'packets_per_sec' in df_iot.columns:
    mmtc_lambda = df_iot.groupby(iot_sec)['packets_per_sec'].sum().reset_index(name='mmtc_lambda')
else:
    mmtc_lambda = df_iot.groupby(iot_sec).size().reset_index(name='mmtc_lambda')
mmtc_lambda.columns = ['sec', 'mmtc_lambda']

print("3. Aligning timeline and merging traffic streams...")

# Sequential index matching
embb_lambda['step'] = range(len(embb_lambda))
urllc_lambda['step'] = range(len(urllc_lambda))
mmtc_lambda['step'] = range(len(mmtc_lambda))

hybrid_df = pd.merge(embb_lambda[['step', 'embb_lambda']], urllc_lambda[['step', 'urllc_lambda']], on='step', how='outer')
hybrid_df = pd.merge(hybrid_df, mmtc_lambda[['step', 'mmtc_lambda']], on='step', how='outer')

hybrid_df.fillna(0, inplace=True)
hybrid_df.rename(columns={'step': 'timestamp'}, inplace=True)

output_path = 'data/final_hybrid_5g_traffic.csv'
hybrid_df.to_csv(output_path, index=False)

print(f"\nSUCCESS: Integrated dataset created at '{output_path}'!")
print("\nFirst 5 Rows of Processed Data:")
print(hybrid_df.head())