import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from nixtla import NixtlaClient
from datetime import datetime

# Set page config for a premium look
st.set_page_config(
    page_title="Mall Sales Anomaly Detection",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium "Dark Mode" aesthetic for KPI cards
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    /* Force visibility of Label and Value */
    div[data-testid="stMetricLabel"] > div {
        color: #9ca3af !important; /* Muted gray for label */
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: #ffffff !important; /* Pure white for value */
        font-size: 32px !important;
    }
    /* Insight Card Styling */
    .insight-card {
        background-color: #1f2937;
        border-left: 5px solid #e74c3c;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        color: #f3f4f6;
    }
    .insight-title {
        color: #9ca3af;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .insight-value {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .insight-desc {
        font-size: 1rem;
        color: #d1d5db;
    }
    .highlight-red { color: #f87171; font-weight: bold; }
    .highlight-yellow { color: #fbbf24; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def load_env():
    """Manually load .env file if it exists."""
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    os.environ[key.strip()] = value.strip()

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    # Default fallback
    dataset_path = "mall_sales.csv"
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return None

@st.cache_data
def preprocess_data(df, ds_col, y_col, id_col):
    # Standardize names for internal logic
    working_df = df[[ds_col, y_col, id_col]].copy()
    working_df = working_df.rename(columns={ds_col: 'ds', y_col: 'y', id_col: 'unique_id'})
    working_df['ds'] = pd.to_datetime(working_df['ds'])
    
    # Filter for the last 6 months (180 days) of data
    max_date = working_df['ds'].max()
    cutoff_date = max_date - pd.Timedelta(days=180)
    
    # Create master date range for padding (last 180 days)
    master_dates = pd.date_range(start=cutoff_date, end=max_date, freq='D')
    master_df = pd.DataFrame({'ds': master_dates})
    
    # Daily aggregation
    daily_sales = working_df.groupby(['unique_id', working_df['ds'].dt.date]).agg({'y': 'sum'}).reset_index()
    daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
    
    # Fill gaps and ENSURE 180 days of history for every ID (Padding)
    all_ids = daily_sales['unique_id'].unique()
    dfs = []
    for uid in all_ids:
        # Get data for this ID
        uid_data = daily_sales[daily_sales['unique_id'] == uid].copy()
        
        # Merge with master date range to ensure full 180-day window
        uid_padded = master_df.merge(uid_data, on='ds', how='left')
        uid_padded['unique_id'] = uid
        uid_padded['y'] = uid_padded['y'].fillna(0)
        
        dfs.append(uid_padded)
    
    return pd.concat(dfs).sort_values(['unique_id', 'ds'])

def get_business_tip(category, direction, magnitude):
    """Generate a specific business strategy based on category and anomaly type."""
    cat = str(category).lower()
    
    # General strategies if category not specifically matched
    if direction == "Spike":
        if "beauty" in cat or "cosmetic" in cat:
            return "🚀 **Opportunity:** Viral trend or influencer mention? Check social mentions and ensure shelf availability."
        if "electronic" in cat or "tech" in cat:
            return "🔋 **Opportunity:** New product launch or tech event? Consider bundling accessories for this high-demand period."
        if "clothing" in cat or "fashion" in cat:
            return "👕 **Opportunity:** Seasonal shift? Double down on similar styles in your upcoming marketing campaign."
        return f"🌟 **Opportunity:** Unusual {magnitude:.1f}x growth! Analyze marketing channels and consider extending current promotions."
    else: # Dip
        if "beauty" in cat or "cosmetic" in cat:
            return "⚠️ **Risk:** Stock-out or competitor sale? Verify inventory levels and check for regional price wars."
        if "electronic" in cat or "tech" in cat:
            return "🔌 **Risk:** Supply chain delay? Monitor shipping logs and provide proactive updates to waiting customers."
        if "clothing" in cat or "fashion" in cat:
            return "👗 **Risk:** Out-of-season inventory? Consider a flash clearance to move slow-moving stock."
        return f"🚨 **Risk:** Sales dropped significantly below expected levels. Check for data entry errors or operational disruptions."

def detect_anomalies(df, level, selected_items, api_key):
    nixtla_client = NixtlaClient(api_key=api_key)
    
    # Filter for selected items
    subset_df = df[df['unique_id'].isin(selected_items)]
    
    if subset_df.empty:
        return pd.DataFrame()
        
    anomalies_df = nixtla_client.detect_anomalies(
        df=subset_df,
        freq='D',
        level=level
    )
    return anomalies_df

def generate_forecast(df, selected_items, api_key, horizon=7):
    nixtla_client = NixtlaClient(api_key=api_key)
    
    subset_df = df[df['unique_id'].isin(selected_items)]
    if subset_df.empty:
        return pd.DataFrame()
        
    forecast_df = nixtla_client.forecast(
        df=subset_df,
        h=horizon,
        freq='D',
        level=[80, 90] # Forecasting intervals
    )
    return forecast_df

def main():
    st.title("🌐 Universal AI Time-Series Explorer")
    st.markdown("### Zero-Shot Anomaly Detection & Forecasting via Nixtla TimeGPT")
    
    # Sidebar: API Key Configuration
    st.sidebar.header("🔑 API Configuration")
    load_env() # Load default from .env if exists
    default_key = os.environ.get("NIXTLA_API_KEY", "")
    nixtla_api_key = st.sidebar.text_input("Nixtla API Key", value=default_key, type="password", help="Get your key at dashboard.nixtla.io")

    # Sidebar: File Upload
    st.sidebar.header("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type="csv")
    
    df = load_data(uploaded_file)
    
    if df is None:
        st.info("👋 Welcome! Please upload a CSV file or ensure `mall_sales.csv` is in the directory to start.")
        return

    # Sidebar: Column Mapping
    st.sidebar.header("🎯 Column Mapping")
    cols = df.columns.tolist()
    
    # Smart defaults for Mall Sales or Retail Sales
    def_ds = "invoice_date" if "invoice_date" in cols else ("Date" if "Date" in cols else cols[0])
    def_y = "price" if "price" in cols else ("Total Amount" if "Total Amount" in cols else cols[1])
    def_id = "shopping_mall" if "shopping_mall" in cols else ("Product Category" if "Product Category" in cols else cols[2])

    ds_col = st.sidebar.selectbox("Date Column (ds)", cols, index=cols.index(def_ds))
    y_col = st.sidebar.selectbox("Value Column (y)", cols, index=cols.index(def_y))
    id_col = st.sidebar.selectbox("Category/ID Column (unique_id)", cols, index=cols.index(def_id))

    # Pre-calculate date range for display
    temp_df = df.copy()
    temp_df[ds_col] = pd.to_datetime(temp_df[ds_col])
    max_date = temp_df[ds_col].max()
    min_date = max_date - pd.Timedelta(days=180)
    st.info(f"📅 **Active Analysis Period:** {min_date.strftime('%b %Y')} to {max_date.strftime('%b %Y')} (Recent 6 Months)")

    daily_sales = preprocess_data(df, ds_col, y_col, id_col)

    # Sidebar: Analysis Controls
    st.sidebar.header("⚙️ Analysis Controls")
    all_items = sorted(daily_sales['unique_id'].unique().tolist())
    selected_items = st.sidebar.multiselect(f"Select {id_col}(s)", all_items, default=all_items[:2] if all_items else [])
    
    sensitivity = st.sidebar.slider("Anomaly Sensitivity", min_value=90.0, max_value=99.9, value=99.0, step=0.1)
    show_forecast = st.sidebar.toggle("Show 7-Day Forecast", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("This tool is dataset-agnostic. Just map your columns and let the AI do the rest.")

    if not nixtla_api_key:
        st.warning("⚠️ Please enter your Nixtla API Key in the sidebar to begin analysis.")
        return

    if not selected_items:
        st.warning(f"Please select at least one {id_col} to begin.")
        return

    with st.spinner("AI is analyzing patterns..."):
        try:
            # 1. Detect Anomalies
            anomalies_df = detect_anomalies(daily_sales, sensitivity, selected_items, nixtla_api_key)
            merged_df = daily_sales.merge(anomalies_df, on=['unique_id', 'ds'], how='inner', suffixes=('', '_anomaly'))
            
            # 2. Generate Forecast if enabled
            forecast_df = None
            if show_forecast:
                forecast_df = generate_forecast(daily_sales, selected_items, nixtla_api_key)

            # KPI Metrics
            cols = st.columns(3)
            total_sales = merged_df[merged_df['unique_id'].isin(selected_items)]['y'].sum()
            total_anomalies = merged_df['anomaly'].sum()
            
            cols[0].metric("Total Sales Volume", f"${total_sales:,.0f}")
            cols[1].metric("Anomalies Detected", int(total_anomalies))
            
            if forecast_df is not None:
                next_week_sales = forecast_df['TimeGPT'].sum()
                cols[2].metric("Projected Week Sales", f"${next_week_sales:,.0f}")
            else:
                cols[2].metric(f"Active {id_col}s", len(selected_items))

            # Visualizations
            st.markdown("---")
            for item in selected_items:
                item_data = merged_df[merged_df['unique_id'] == item]
                
                fig = go.Figure()
                
                # Historical Data
                fig.add_trace(go.Scatter(
                    x=item_data['ds'], y=item_data['y'],
                    mode='lines',
                    name='Actual History',
                    line=dict(color='#3498db', width=2),
                    opacity=0.7
                ))
                
                # Anomalies (Red 'X' markers on top of the line)
                anoms = item_data[item_data['anomaly'] == 1]
                fig.add_trace(go.Scatter(
                    x=anoms['ds'], y=anoms['y'],
                    mode='markers',
                    name='🚨 AI Flagged Anomaly',
                    marker=dict(color='#e74c3c', size=10, symbol='x'),
                    hovertemplate="<b>🚨 ANOMALY DETECTED</b><br>Date: %{x}<extra></extra>"
                ))

                # Forecast
                if forecast_df is not None:
                    item_forecast = forecast_df[forecast_df['unique_id'] == item]
                    
                    # Connect history to forecast
                    last_hist = item_data.iloc[-1]
                    f_ds = pd.concat([pd.Series([last_hist['ds']]), item_forecast['ds']])
                    f_y = pd.concat([pd.Series([last_hist['y']]), item_forecast['TimeGPT']])

                    fig.add_trace(go.Scatter(
                        x=f_ds, y=f_y,
                        mode='lines',
                        name='AI Projection',
                        line=dict(color='#2ecc71', width=3, dash='dash'),
                    ))
                    
                    # Confidence Intervals
                    fig.add_trace(go.Scatter(
                        x=item_forecast['ds'], y=item_forecast['TimeGPT-hi-90'],
                        mode='lines', line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=item_forecast['ds'], y=item_forecast['TimeGPT-lo-90'],
                        mode='lines', fill='tonexty',
                        fillcolor='rgba(46, 204, 113, 0.1)',
                        line=dict(width=0), name='90% Confidence', showlegend=False
                    ))

                fig.update_layout(
                    title=f"Time-Series Analysis: {item}",
                    xaxis_title="Date",
                    yaxis_title=y_col,
                    template="plotly_white",
                    height=500,
                    margin=dict(l=20, r=20, t=60, b=20),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # Detailed Insight Cards
            if total_anomalies > 0:
                st.markdown("### 💡 Automated Business Insights")
                anomalous_data = merged_df[merged_df['anomaly'] == 1].sort_values('ds', ascending=False)
                
                # We show the top 6 most recent or significant anomalies as cards
                display_cols = st.columns(2)
                for i, (_, row) in enumerate(anomalous_data.head(6).iterrows()):
                    col_idx = i % 2
                    with display_cols[col_idx]:
                        # 1. Determine "Why": Spike vs Dip
                        # More robust column finding: look for columns containing 'hi' or 'lo'
                        hi_cols = [c for c in anomalous_data.columns if 'hi' in c.lower()]
                        lo_cols = [c for c in anomalous_data.columns if 'lo' in c.lower()]
                        
                        if not hi_cols or not lo_cols:
                            continue # Should not happen, but prevents crash
                            
                        upper_bound = row[hi_cols[0]]
                        lower_bound = row[lo_cols[0]]
                        actual = row['y']
                        
                        if actual > upper_bound:
                            direction = "Spike"
                            magnitude = actual / upper_bound if upper_bound > 0 else (actual/1.0)
                            status_icon = "🚀"
                            status_text = "Positive Spike"
                            color_class = "highlight-red" # Using red for attention as requested
                            border_color = "#2ecc71" # Green border for growth
                        else:
                            direction = "Dip"
                            magnitude = lower_bound / actual if actual > 0 else 0
                            status_icon = "📉"
                            status_text = "Negative Dip"
                            color_class = "highlight-red" # Red for risk
                            border_color = "#e74c3c" # Red border for loss

                        # 2. Get tailored strategy
                        strategy = get_business_tip(row['unique_id'], direction, magnitude)

                        st.markdown(f"""
                        <div class="insight-card" style="border-left-color: {border_color}">
                            <div class="insight-title">{row['ds'].strftime('%B %d, %Y')} | {row['unique_id']}</div>
                            <div class="insight-value">{status_icon} {status_text}</div>
                            <div class="insight-desc">
                                Actual value of <span class="{color_class}">{actual:,.2f}</span> was 
                                <span class="{color_class}">{magnitude:.1f}x</span> {"higher" if direction == "Spike" else "lower"} than the AI predicted.
                                <br><br>
                                <b>💡 Strategy:</b> {strategy}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if len(anomalous_data) > 6:
                    st.info(f"Showing top 6 of {len(anomalous_data)} total anomalies. Adjust sensitivity to refine results.")
            else:
                st.success(f"No anomalies detected for {selected_items} at the current sensitivity level.")

        except Exception as e:
            st.error(f"Error during analysis: {e}")
            if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                st.warning("Please check your Nixtla API Key in the sidebar.")

if __name__ == "__main__":
    main()
