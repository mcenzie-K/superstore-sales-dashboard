
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

st.sidebar.title("Navigation")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    return df

df = load_data()

# Feature Engineering (consistent with notebook)
def feature_engineer_df(df_input):
    df_output = df_input.copy()
    df_output['Order Year'] = df_output['Order Date'].dt.year
    df_output['Order Month'] = df_output['Order Date'].dt.month
    df_output['Order Week Number'] = df_output['Order Date'].dt.isocalendar().week.astype(int)
    df_output['Order Day of Week'] = df_output['Order Date'].dt.dayofweek
    df_output['Order Quarter'] = df_output['Order Date'].dt.quarter
    return df_output

df_processed = feature_engineer_df(df.copy())

# Aggregate monthly sales for overall forecasting
monthly_sales = df_processed.groupby(['Order Year', 'Order Month'])['Sales'].sum().reset_index()
monthly_sales.rename(columns={'Sales': 'Monthly Sales'}, inplace=True)
monthly_sales['Order_Period'] = pd.to_datetime(monthly_sales['Order Year'].astype(str) + '-' + monthly_sales['Order Month'].astype(str) + '-01')
monthly_sales = monthly_sales.set_index('Order_Period')
monthly_sales.index = pd.to_datetime(monthly_sales.index)
monthly_sales = monthly_sales.asfreq('MS')

# Prepare weekly sales for anomaly detection
daily_sales_ts = df_processed.groupby(pd.Grouper(key='Order Date', freq='D'))['Sales'].sum().sort_index()
weekly_sales_ts = daily_sales_ts.resample('W-MON').sum()
weekly_sales_ts = weekly_sales_ts[weekly_sales_ts > 0].dropna()


# --- Forecasting Models (cached) ---
@st.cache_resource
def train_overall_xgboost(monthly_sales_df):
    xgb_df = monthly_sales_df.copy()
    xgb_df['Sales_Lag_1'] = xgb_df['Monthly Sales'].shift(1)
    xgb_df['Sales_Lag_2'] = xgb_df['Monthly Sales'].shift(2)
    xgb_df['Sales_Lag_3'] = xgb_df['Monthly Sales'].shift(3)
    xgb_df['Rolling_Mean_3'] = xgb_df['Monthly Sales'].rolling(window=3).mean()
    xgb_df['Month'] = xgb_df.index.month
    xgb_df['Year'] = xgb_df.index.year
    xgb_df['Quarter'] = xgb_df.index.quarter
    xgb_df.dropna(inplace=True)

    X = xgb_df.drop('Monthly Sales', axis=1)
    y = xgb_df['Monthly Sales']

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X, y)
    return xgb_model, xgb_df

overall_xgb_model, overall_xgb_df = train_overall_xgboost(monthly_sales.copy())


@st.cache_resource
def train_segment_xgboost(df_original, segment_type, segment_value):
    # Filter data for the specific segment
    if segment_type == 'Category':
        filtered_df = df_original[df_original['Category'] == segment_value].copy()
    elif segment_type == 'Region':
        filtered_df = df_original[df_original['Region'] == segment_value].copy()
    else:
        raise ValueError("segment_type must be 'Category' or 'Region'")

    # Aggregate to monthly sales
    monthly_segment_sales = filtered_df.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
    monthly_segment_sales.rename(columns={'Order Date': 'Order_Period', 'Sales': 'Monthly Sales'}, inplace=True)
    monthly_segment_sales = monthly_segment_sales.set_index('Order_Period')
    monthly_segment_sales.index = pd.to_datetime(monthly_segment_sales.index)

    # Create lag features
    monthly_segment_sales['Sales_Lag_1'] = monthly_segment_sales['Monthly Sales'].shift(1)
    monthly_segment_sales['Sales_Lag_2'] = monthly_segment_sales['Monthly Sales'].shift(2)
    monthly_segment_sales['Sales_Lag_3'] = monthly_segment_sales['Monthly Sales'].shift(3)

    # Create 3-month rolling mean
    monthly_segment_sales['Rolling_Mean_3'] = monthly_segment_sales['Monthly Sales'].rolling(window=3).mean()

    # Add time-based features
    monthly_segment_sales['Month'] = monthly_segment_sales.index.month
    monthly_segment_sales['Year'] = monthly_segment_sales.index.year
    monthly_segment_sales['Quarter'] = monthly_segment_sales.index.quarter

    # Drop rows with NaN values resulting from lag/rolling mean creation
    monthly_segment_sales.dropna(inplace=True)

    if monthly_segment_sales.empty: # Handle cases where segment has no data after lags
        return None, None

    X_seg = monthly_segment_sales.drop('Monthly Sales', axis=1)
    y_seg = monthly_segment_sales['Monthly Sales']

    xgb_model_seg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model_seg.fit(X_seg, y_seg)
    return xgb_model_seg, monthly_segment_sales


product_categories = ['Furniture', 'Technology', 'Office Supplies']
regions = ['West', 'East', 'Central', 'South'] # Include all regions

segment_models = {}
segment_data_frames = {}

for category in product_categories:
    model, data = train_segment_xgboost(df_processed, 'Category', category)
    if model and data is not None:
        segment_models[f'Category: {category}'] = model
        segment_data_frames[f'Category: {category}'] = data

for region in regions:
    model, data = train_segment_xgboost(df_processed, 'Region', region)
    if model and data is not None:
        segment_models[f'Region: {region}'] = model
        segment_data_frames[f'Region: {region}'] = data


# --- Clustering Data (cached) ---
@st.cache_resource
def prepare_clustering_data(df_input):
    # Ensure 'Order Date' is datetime and sort the DataFrame by it
    df_input['Order Date'] = pd.to_datetime(df_input['Order Date'], dayfirst=True)
    df_input = df_input.sort_values(by='Order Date')

    # Calculate total sales volume per sub-category
    total_sales_volume = df_input.groupby('Sub-Category')['Sales'].sum().rename('Total Sales Volume')

    # Calculate monthly sales for each sub-category for volatility and growth rate
    monthly_subcategory_sales = df_input.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='MS')])['Sales'].sum().unstack(fill_value=0)

    # Calculate Sales Volatility (standard deviation of monthly sales)
    sales_volatility = monthly_subcategory_sales.std(axis=1).rename('Sales Volatility')

    # Calculate Year-over-Year Sales Growth Rate
    year_sales = df_input.groupby(['Sub-Category', df_input['Order Date'].dt.year])['Sales'].sum().unstack(fill_value=0)
    latest_year = df_input['Order Date'].dt.year.max()
    second_latest_year = latest_year - 1

    def calculate_yoy_growth(row):
        sales_latest = row.get(latest_year, 0)
        sales_second_latest = row.get(second_latest_year, 0)
        if sales_second_latest > 0:
            return ((sales_latest - sales_second_latest) / sales_second_latest) * 100
        return 0

    sales_growth_rate = year_sales.apply(calculate_yoy_growth, axis=1).rename('Sales Growth Rate')

    # Calculate Average Order Value
    df_input['Order Total'] = df_input.groupby('Order ID')['Sales'].transform('sum')
    average_order_value = df_input.groupby('Sub-Category')['Order Total'].mean().rename('Average Order Value')

    # Combine all features into a single DataFrame for clustering
    clustering_df = pd.DataFrame({
        'Total Sales Volume': total_sales_volume,
        'Sales Growth Rate': sales_growth_rate,
        'Sales Volatility': sales_volatility,
        'Average Order Value': average_order_value
    }).dropna()

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_df)
    scaled_clustering_df = pd.DataFrame(scaled_features, columns=clustering_df.columns, index=clustering_df.index)

    # Apply KMeans (optimal_k=3 from notebook)
    optimal_k = 3
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clustering_df['Cluster'] = kmeans_final.fit_predict(scaled_clustering_df)

    # Label clusters (consistent with notebook)
    cluster_labels_map = {
        0: 'High Growth & Volume',
        1: 'Stable Low Demand',
        2: 'Declining & Volatile'
    }
    clustering_df['Demand Segment'] = clustering_df['Cluster'].map(cluster_labels_map)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_clustering_df)
    pca_df = pd.DataFrame(data=pca_components, columns=['Principal Component 1', 'Principal Component 2'], index=clustering_df.index)
    pca_df['Cluster'] = clustering_df['Cluster']
    pca_df['Demand Segment'] = clustering_df['Demand Segment']

    return clustering_df, pca_df, scaler, kmeans_final

clustering_df, pca_df, clustering_scaler, clustering_kmeans = prepare_clustering_data(df.copy())


# --- HELPER FUNCTIONS FOR FORECASTING ---
def get_next_month_features(last_date, current_df, model_features, forecast_horizon):
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
    future_data = []

    temp_df = current_df.copy().set_index('Order_Period') if 'Order_Period' in current_df.columns else current_df.copy()
    temp_df = temp_df[['Monthly Sales', 'Month', 'Year', 'Quarter']]

    for i in range(forecast_horizon):
        current_date = future_dates[i]
        last_month_sales = temp_df['Monthly Sales'].iloc[-1] if not temp_df.empty else 0
        last_2month_sales = temp_df['Monthly Sales'].iloc[-2] if len(temp_df) >= 2 else 0
        last_3month_sales = temp_df['Monthly Sales'].iloc[-3] if len(temp_df) >= 3 else 0

        # Ensure rolling mean calculation is robust for short data
        rolling_mean_3 = temp_df['Monthly Sales'].rolling(window=min(3, len(temp_df))).mean().iloc[-1] if not temp_df.empty else 0

        # Create a new row of features for the current future date
        new_features = {
            'Sales_Lag_1': last_month_sales,
            'Sales_Lag_2': last_2month_sales,
            'Sales_Lag_3': last_3month_sales,
            'Rolling_Mean_3': rolling_mean_3,
            'Month': current_date.month,
            'Year': current_date.year,
            'Quarter': current_date.quarter
        }
        future_df_row = pd.DataFrame([new_features], index=[current_date])

        # Predict for this single future point
        next_month_prediction = overall_xgb_model.predict(future_df_row[model_features])

        # Append the prediction to the temporary DataFrame to be used as lag for the next forecast step
        temp_df = pd.concat([temp_df, pd.DataFrame({'Monthly Sales': next_month_prediction[0], 'Month': current_date.month, 'Year': current_date.year, 'Quarter': current_date.quarter}, index=[current_date])])

        future_data.append({
            'Order_Period': current_date,
            'Monthly Sales': next_month_prediction[0]
        })

    return pd.DataFrame(future_data).set_index('Order_Period')


def get_segment_next_month_features(last_date, segment_data, model, model_features, forecast_horizon):
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
    future_data = []

    temp_df = segment_data.copy()

    for i in range(forecast_horizon):
        current_date = future_dates[i]
        
        last_month_sales = temp_df['Monthly Sales'].iloc[-1] if not temp_df.empty else 0
        last_2month_sales = temp_df['Monthly Sales'].iloc[-2] if len(temp_df) >= 2 else 0
        last_3month_sales = temp_df['Monthly Sales'].iloc[-3] if len(temp_df) >= 3 else 0

        rolling_mean_3 = temp_df['Monthly Sales'].rolling(window=min(3, len(temp_df))).mean().iloc[-1] if not temp_df.empty else 0

        new_features = {
            'Sales_Lag_1': last_month_sales,
            'Sales_Lag_2': last_2month_sales,
            'Sales_Lag_3': last_3month_sales,
            'Rolling_Mean_3': rolling_mean_3,
            'Month': current_date.month,
            'Year': current_date.year,
            'Quarter': current_date.quarter
        }
        future_df_row = pd.DataFrame([new_features], index=[current_date])
        
        # Ensure the feature columns match what the model was trained on
        next_month_prediction = model.predict(future_df_row[model_features])
        
        # Append the prediction to the temporary DataFrame
        temp_df = pd.concat([temp_df, pd.DataFrame({
            'Monthly Sales': next_month_prediction[0],
            'Sales_Lag_1': last_month_sales, # These will be re-calculated in the next iteration if needed
            'Sales_Lag_2': last_2month_sales,
            'Sales_Lag_3': last_3month_sales,
            'Rolling_Mean_3': rolling_mean_3,
            'Month': current_date.month,
            'Year': current_date.year,
            'Quarter': current_date.quarter
        }, index=[current_date])])
        
        future_data.append({
            'Order_Period': current_date,
            'Monthly Sales': next_month_prediction[0]
        })

    return pd.DataFrame(future_data).set_index('Order_Period')


# --- Streamlit App Pages ---
def sales_overview_dashboard():
    st.header("Sales Overview Dashboard")

    # Total Sales by Year (Bar Chart)
    yearly_sales = df_processed.groupby('Order Year')['Sales'].sum().reset_index()
    fig_yearly_sales = px.bar(yearly_sales, x='Order Year', y='Sales', title='Total Sales by Year')
    st.plotly_chart(fig_yearly_sales, use_container_width=True)

    # Monthly Sales Trend (Line Chart)
    fig_monthly_sales = px.line(monthly_sales.reset_index(), x='Order_Period', y='Monthly Sales', title='Overall Monthly Sales Trend (2015-2018)')
    st.plotly_chart(fig_monthly_sales, use_container_width=True)

    st.subheader("Sales by Region and Category")

    # Filters
    selected_region = st.selectbox("Select Region", ['All'] + list(df_processed['Region'].unique()))
    selected_category = st.selectbox("Select Category", ['All'] + list(df_processed['Category'].unique()))

    filtered_df = df_processed.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    # Group by Region and Category to ensure all combinations appear
    sales_by_region_category = filtered_df.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    
    if not sales_by_region_category.empty:
        fig_region_category = px.bar(sales_by_region_category,
                                 x='Region', y='Sales', color='Category',
                                 title=f'Sales by Region and Category (Filtered)',
                                 labels={'Sales': 'Total Sales Amount'})
        st.plotly_chart(fig_region_category, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

def forecast_explorer():
    st.header("Forecast Explorer (XGBoost Model)")

    # Selector for Overall vs. Segment Forecast
    forecast_type = st.radio("Select Forecast Type", ["Overall Sales", "Segment-Specific Sales"])

    if forecast_type == "Overall Sales":
        st.subheader("Overall Monthly Sales Forecast")
        forecast_horizon_overall = st.slider("Select Forecast Horizon (Months)", 1, 3, 3)
        
        last_date_overall = monthly_sales.index.max()
        model_features_overall = list(overall_xgb_df.drop('Monthly Sales', axis=1).columns)

        # Generate forecast
        future_forecast_overall = get_next_month_features(last_date_overall, monthly_sales.reset_index(), model_features_overall, forecast_horizon_overall)

        # Combine historical and forecasted data
        combined_overall = pd.concat([monthly_sales['Monthly Sales'], future_forecast_overall['Monthly Sales']])

        fig_overall_forecast = px.line(combined_overall.reset_index(), x='index', y='Monthly Sales',
                                     title=f'Overall Monthly Sales and {forecast_horizon_overall}-Month Forecast',
                                     labels={'index': 'Date', 'Monthly Sales': 'Sales Amount'})
        fig_overall_forecast.add_vline(x=last_date_overall, line_dash="dash", line_color="red", annotation_text="Forecast Start")
        st.plotly_chart(fig_overall_forecast, use_container_width=True)

        st.subheader("Forecast Details:")
        st.dataframe(future_forecast_overall, use_container_width=True)

    else:
        st.subheader("Segment-Specific Sales Forecast")
        segment_choice = st.selectbox("Select Segment", list(segment_models.keys()))
        forecast_horizon_segment = st.slider("Select Forecast Horizon (Months) for Segment", 1, 3, 3)

        if segment_choice and segment_models[segment_choice] is not None:
            segment_model = segment_models[segment_choice]
            segment_data = segment_data_frames[segment_choice]
            last_date_segment = segment_data.index.max()
            model_features_segment = list(segment_data.drop('Monthly Sales', axis=1).columns)

            # Generate forecast
            future_forecast_segment = get_segment_next_month_features(last_date_segment, segment_data, segment_model, model_features_segment, forecast_horizon_segment)

            # Combine historical and forecasted data
            combined_segment = pd.concat([segment_data['Monthly Sales'], future_forecast_segment['Monthly Sales']])

            fig_segment_forecast = px.line(combined_segment.reset_index(), x='index', y='Monthly Sales',
                                         title=f'{segment_choice} Monthly Sales and {forecast_horizon_segment}-Month Forecast',
                                         labels={'index': 'Date', 'Monthly Sales': 'Sales Amount'})
            fig_segment_forecast.add_vline(x=last_date_segment, line_dash="dash", line_color="red", annotation_text="Forecast Start")
            st.plotly_chart(fig_segment_forecast, use_container_width=True)

            st.subheader("Forecast Details:")
            st.dataframe(future_forecast_segment, use_container_width=True)

            # Metrics (simplified for dashboard, actuals not available for future forecast)
            st.markdown("**Note:** MAE and RMSE require actual future values for calculation. These metrics below are indicative of past model performance, not future accuracy.")


def anomaly_report():
    st.header("Anomaly Report")

    st.write("Detected sales anomalies using Isolation Forest.")

    # Perform anomaly detection (re-run as IsolationForest is not easily cached on predictions)
    X_iso = weekly_sales_ts.values.reshape(-1, 1)
    iso_forest = IsolationForest(random_state=42, contamination='auto')
    iso_forest.fit(X_iso)
    anomaly_predictions_iso = iso_forest.predict(X_iso)
    anomalies_iso = weekly_sales_ts[anomaly_predictions_iso == -1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_sales_ts.index, y=weekly_sales_ts.values, mode='lines', name='Weekly Sales', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=anomalies_iso.index, y=anomalies_iso.values, mode='markers', name='Anomalies', marker=dict(color='red', size=8, symbol='circle')))
    
    fig.update_layout(title='Weekly Sales with Detected Anomalies (Isolation Forest)',
                      xaxis_title='Date', yaxis_title='Weekly Sales')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomaly Dates and Sales Values:")
    if not anomalies_iso.empty:
        st.dataframe(anomalies_iso.rename("Sales"), use_container_width=True)
    else:
        st.info("No anomalies detected.")

def product_demand_segments():
    st.header("Product Demand Segments")

    st.write("Sub-categories clustered into demand segments based on sales volume, growth, volatility, and average order value.")

    # Cluster Chart from PCA
    fig_pca = px.scatter(pca_df, x='Principal Component 1', y='Principal Component 2', color='Demand Segment',
                         title='Product Demand Segments (PCA-Reduced)',
                         hover_name=pca_df.index,
                         color_discrete_map={
                             'High Growth & Volume': 'green',
                             'Stable Low Demand': 'blue',
                             'Declining & Volatile': 'red'
                         })
    st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Sub-Categories by Demand Segment:")
    selected_segment = st.selectbox("Select Demand Segment", ['All'] + list(clustering_df['Demand Segment'].unique()))

    if selected_segment != 'All':
        segment_subcategories = clustering_df[clustering_df['Demand Segment'] == selected_segment]
    else:
        segment_subcategories = clustering_df
    
    st.dataframe(segment_subcategories[['Demand Segment', 'Total Sales Volume', 'Sales Growth Rate', 'Sales Volatility', 'Average Order Value']], use_container_width=True)


# --- Main App Logic ---
page_names_to_funcs = {
    "Sales Overview Dashboard": sales_overview_dashboard,
    "Forecast Explorer": forecast_explorer,
    "Anomaly Report": anomaly_report,
    "Product Demand Segments": product_demand_segments,
}

selected_page = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
pages = list(page_names_to_funcs.keys())

# Dynamically set the selected page based on the sidebar selection
page_names_to_funcs[selected_page]()
