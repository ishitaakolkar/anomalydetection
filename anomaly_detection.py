import pandas as pd
import matplotlib.pyplot as plt
import os
from nixtla import NixtlaClient

def load_env():
    """Manually load .env file if it exists."""
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    os.environ[key.strip()] = value.strip()

def main():
    load_env()
    print("--- 1. Loading the dataset and validating schema ---")
    dataset_path = "mall_sales.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows.")

    print("\n--- 2. Converting ds column to datetime ---")
    # Mapping columns as per the dataset view: invoice_date -> ds, price -> y, shopping_mall -> unique_id
    df = df.rename(columns={'invoice_date': 'ds', 'price': 'y', 'shopping_mall': 'unique_id'})
    df['ds'] = pd.to_datetime(df['ds'])

    print("\n--- 3. Sorting and Aggregating data ---")
    # TimeGPT needs one observation per unique_id and ds.
    daily_sales = df.groupby(['unique_id', df['ds'].dt.date]).agg({'y': 'sum'}).reset_index()
    daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
    
    # Ensure each series (mall) is regular and has no gaps
    all_malls = daily_sales['unique_id'].unique()
    dfs = []
    for mall in all_malls:
        mall_df = daily_sales[daily_sales['unique_id'] == mall].copy()
        mall_df = mall_df.set_index('ds').resample('D').asfreq().reset_index()
        mall_df['unique_id'] = mall
        mall_df['y'] = mall_df['y'].fillna(0)
        dfs.append(mall_df)
    
    daily_sales = pd.concat(dfs).sort_values(['unique_id', 'ds'])
    print(f"Prepared daily_sales with {len(daily_sales)} rows.")
    print(daily_sales.head())

    print("\n--- 4. Performing anomaly detection using TimeGPT ---")
    nixtla_client = NixtlaClient() # Assumes NIXTLA_API_KEY is in env
    
    # Selecting one mall for clear demo visualization (e.g., Kanyon or Zorlu Center)
    # Testing with level=99 (default) or individual integer
    anomalies_df = nixtla_client.detect_anomalies(
        df=daily_sales,
        freq='D',
        level=99
    )
    print("Anomalies calculated.")
    print(f"Anomalies columns: {anomalies_df.columns.tolist()}")

    print("\n--- 5. Merging anomaly flags back ---")
    # If anomalies_df has y, it will create y_x and y_y. Let's handle it.
    merged_df = daily_sales.merge(anomalies_df, on=['unique_id', 'ds'], how='left', suffixes=('', '_anomaly'))
    
    # Ensure 'y' and 'anomaly' are available
    if 'y' not in merged_df.columns and 'y_anomaly' in merged_df.columns:
        merged_df = merged_df.rename(columns={'y_anomaly': 'y'})

    print("\n--- 6. Generating clear visualizations ---")
    # We'll pick one mall to visualize for the demo: "Kanyon"
    target_mall = "Kanyon"
    mall_data = merged_df[merged_df['unique_id'] == target_mall]
    
    plt.figure(figsize=(12, 6))
    plt.plot(mall_data['ds'], mall_data['y'], label='Daily Sales', color='blue', alpha=0.6)
    
    # Highlight anomalies
    anomalies = mall_data[mall_data['anomaly'] == 1]
    plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Detected Anomalies', s=50, marker='x')

    plt.title(f"Mall Sales Anomaly Detection - {target_mall}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales ($)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = "mall_sales_anomalies.png"
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

    print("\n--- 7. Summary Report ---")
    total_anomalies = merged_df['anomaly'].sum()
    print(f"Total anomalies detected across all malls: {total_anomalies}")
    
    print(f"\nTop Anomalies for {target_mall}:")
    top_anomalies = anomalies.sort_values(by='y', ascending=False).head(3)
    for idx, row in top_anomalies.iterrows():
        print(f"  - Date: {row['ds'].strftime('%Y-%m-%d')}, Sales: ${row['y']:.2f}, Interpretation: Significant deviation from normal shopping patterns.")

    print("\n--- 8. Business-Friendly Explanation ---")
    print("These anomalies represent unusual spikes or drops in daily sales that cannot be explained by typical trends.")
    print("For a business stakeholder, these points might correspond to external events, logistics delays, or highly successful local promotions.")

if __name__ == "__main__":
    main()
