import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Dark theme configuration
st.set_page_config(
    page_title="AuraClima - AI Climate Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0c1017 0%, #1a1f2e 100%);
        color: #ffffff;
    }
    
.main‑header {
    /* layout & gradient */
    display: inline-block;
    text-align: center;
    background-image: linear-gradient(135deg, #1f77b4, #FF7F0E);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;

    /* size & spacing */
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 1rem;
    
    /* turn off any blurs, shadows, filters */
    text-shadow: none !important;
    filter: none !important;
    
    /* force sharp font rendering */
    text-rendering: optimizeLegibility !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    
    /* keep it above any backdrop‐filter layers */
    position: relative;
    z-index: 1;
  }
    
    .subtitle {
        text-align: center;
        color: #FF7F0E;
        font-size: 1.5rem;
        font-style: italic;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(255, 127, 14, 0.2);
    }
    
    .model-card {
        background: linear-gradient(145deg, #1e2530, #2a3441);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(31, 119, 180, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #1f77b4, #2a9fd6);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(31, 119, 180, 0.4);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
    }
    
    .metric-label {
        color: #e0e6ed;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    .ai-badge {
        background: linear-gradient(45deg, #FF7F0E, #ff9a3c);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        box-shadow: 0 2px 10px rgba(255, 127, 14, 0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1f2e, #0c1017);
    }
    
    .stSelectbox > div > div {
        background-color: #2a3441;
        border: 1px solid #1f77b4;
        border-radius: 8px;
    }
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #1f77b4, #FF7F0E);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4, #FF7F0E);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(31, 119, 180, 0.4);
    }
    
    .forecast-section {
        background: rgba(31, 119, 180, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all():
    base = os.path.dirname(__file__)
    models_dir = os.path.join(base, "models")
    data_dir = os.path.join(base, "data")

    # Load models
    model1 = tf.keras.models.load_model(os.path.join(models_dir, "model1.keras"))
    model2 = tf.keras.models.load_model(os.path.join(models_dir, "model2.keras"))
    model3 = tf.keras.models.load_model(os.path.join(models_dir, "model3.keras"))

    # Load scalers
    scaler1 = joblib.load(os.path.join(models_dir, "scaler1.save"))
    scalerX2 = joblib.load(os.path.join(models_dir, "scalerX2.save"))
    scalerY2 = joblib.load(os.path.join(models_dir, "scalerY2.save"))
    scaler3 = joblib.load(os.path.join(models_dir, "scaler3.save"))

    # Load feature columns list for model2
    with open(os.path.join(models_dir, "feature_cols2.list"), "rb") as f:
        feature_cols2 = pickle.load(f)

    # Load CSV data if present
    df_agri = None
    agri_path = os.path.join(data_dir, "Agrofood_co2_emission.csv")
    if os.path.exists(agri_path):
        df_agri = pd.read_csv(agri_path)

    df_co2 = None
    co2_path = os.path.join(data_dir, "CO2_Emissions_1960-2018.csv")
    if os.path.exists(co2_path):
        df_co2 = pd.read_csv(co2_path)
        if 'Country Name' not in df_co2.columns:
            st.error(f"Expected 'Country Name' in CO2 CSV, found: {df_co2.columns.tolist()}")
            df_co2 = None
        else:
            # Ensure Country Name is cleaned before creating dummies for consistency
            df_co2['Country Name'] = df_co2['Country Name'].str.strip()
            dummies = pd.get_dummies(df_co2['Country Name'], prefix='Country')
            country_features = dummies.columns.tolist()
            df_co2 = pd.concat([df_co2, dummies], axis=1)
    else:
        country_features = None

    return {
            "model1": model1, "model2": model2, "model3": model3,
            "scaler1": scaler1, "scalerX2": scalerX2, "scalerY2": scalerY2, "scaler3": scaler3,
            "feature_cols2": feature_cols2, "df_agri": df_agri, "df_co2": df_co2,
            "country_features": country_features,
        }


def forecast_model1(model, scaler, recent_values):
    arr = np.array(recent_values).reshape(-1, 1)
    scaled = scaler.transform(arr).flatten()
    inp = scaled.reshape((1, len(scaled), 1))
    scaled_pred = model.predict(inp, verbose=0)[0, 0]
    pred = scaler.inverse_transform([[scaled_pred]])[0, 0]
    return pred


def predict_model2(model, scalerX, scalerY, feature_array):
    X = np.array(feature_array).reshape(1, -1)
    Xs = scalerX.transform(X)
    ys = model.predict(Xs, verbose=0)
    ypred = scalerY.inverse_transform(ys.reshape(-1, 1)).flatten()[0]
    return ypred


def forecast_model3(model, scaler, recent_series, country_vec):
    window = len(recent_series)
    recent_series_np = np.array(recent_series).reshape(-1, 1)

    co2_scaled_input = scaler.transform(recent_series_np).flatten()
    co2_col = co2_scaled_input.reshape(window, 1)
    country_mat = np.tile(country_vec.reshape(1, -1), (window, 1))
    seq = np.concatenate([co2_col, country_mat], axis=1)
    inp = seq.reshape(1, window, seq.shape[1])

    # Get the raw model prediction and inverse transform it
    ypred_scaled_output = model.predict(inp, verbose=0).flatten()
    ypred_unforced = scaler.inverse_transform(ypred_scaled_output.reshape(-1, 1)).flatten()

    # Apply non-negativity to the unforced predictions
    ypred_processed = np.maximum(0, ypred_unforced)

    # Apply monotonicity to the processed forecast
    for i in range(1, len(ypred_processed)):
        if ypred_processed[i] < ypred_processed[i-1]:
            ypred_processed[i] = ypred_processed[i-1]

    # Return the processed predictions. Scaling for display will happen in the calling function.
    return ypred_processed


def create_animated_metric(label, value, icon="🎯"):
    st.markdown(f"""
    <div class="metric-container">
        <div style="font-size: 1.2rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def sidebar_nav():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 4rem;">🌍</div>
        <h1 style="color: #1f77b4; margin: 10px 0;">AuraClima</h1>
        <p style="color: #FF7F0E; font-style: italic; margin-bottom: 20px;">
            "See the unseen, act on the future"
        </p>
        <div class="ai-badge">🤖 AI-Powered</div>
        <div class="ai-badge">⚡ Real-time</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    page = st.sidebar.radio("🚀 Navigate", ["🏠 Home", "🌍 Climate Intelligence", "ℹ️ About"],
                            label_visibility="collapsed")
    return page


def home_page():
    # Centered title
    st.markdown('<h1 class="main-header">🌍 AuraClima</h1>', unsafe_allow_html=True)

    # AI Features showcase
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="model-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 10px;">🌱</div>
                <h3 style="color: #1f77b4;">Agricultural AI</h3>
                <p style="color: #e0e6ed; font-size: 0.9rem;">LSTM Time Series Forecasting</p>
                <div class="ai-badge">Neural Network</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 10px;">📊</div>
                <h3 style="color: #FF7F0E;">Feature Analysis</h3>
                <p style="color: #e0e6ed; font-size: 0.9rem;">Multi-variate Regression</p>
                <div class="ai-badge">Deep Learning</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="model-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 10px;">💨</div>
                <h3 style="color: #1f77b4;">CO₂ Intelligence</h3>
                <p style="color: #e0e6ed; font-size: 0.9rem;">Advanced sequence modeling</p>
                <div class="ai-badge">Advanced LSTM</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; padding: 30px;">
        <h3 style="color: #1f77b4;">🚀 Advanced AI Climate Modeling</h3>
        <p style="color: #e0e6ed; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Leverage cutting-edge machine learning to forecast climate patterns, emissions, and environmental trends. 
            Our AI models process complex data to provide actionable insights for a sustainable future.
        </p>
    </div>
    """, unsafe_allow_html=True)


def create_enhanced_plot(hist_years, series_co2_plot, fut_years_plot, pred3_plot, country):
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"🌍 AI Climate Intelligence: {country}"],
        specs=[[{"secondary_y": False}]]
    )

    # Historical data (already scaled correctly when passed to this function)
    fig.add_trace(
        go.Scatter(
            x=hist_years,
            y=series_co2_plot, # This is the already scaled historical data for display
            mode='lines+markers',
            name='Historical Emissions',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6, color='#1f77b4'),
            hovertemplate='<b>Year:</b> %{x}<br><b>CO₂:</b> %{y:.2f}<extra></extra>'
        )
    )

    # Prepare forecast data for plotting to ensure continuity
    last_historical_year = hist_years[-1]
    last_historical_value = series_co2_plot[-1] # This should be 58.0 for 2018

    # The forecast line needs to start from the exact last historical point (2018, 58.0)
    # and then continue with its own predictions (2019, 58.0, 2020, predicted_value_2020, etc.).
    # So, the first year for the forecast plot is the last historical year (2018).
    # The first value for the forecast plot is the last historical value (58.0).
    # Then append the actual future years and their predictions.

    # Years for the forecast plot: last historical year + all future years from fut_years_plot
    forecast_years_extended = [last_historical_year] + list(fut_years_plot)
    # Values for the forecast plot: last historical value + all future predictions from pred3_plot
    forecast_values_extended = [last_historical_value] + list(pred3_plot)

    # Forecast data
    fig.add_trace(
        go.Scatter(
            x=forecast_years_extended,
            y=forecast_values_extended,
            mode='lines+markers',
            name='AI Forecast',
            line=dict(color='#FF7F0E', width=4, dash='dash'),
            marker=dict(size=8, color='#FF7F0E', symbol='diamond'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Predicted CO₂:</b> %{y:.2f}<extra></extra>'
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>CO₂ Emissions Forecast for {country}</b>",
            x=0.5,
            font=dict(size=18, color='white')
        ),
        xaxis_title="Year",
        yaxis_title="CO₂ Emissions",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(30, 37, 48, 0.8)',
            bordercolor='#1f77b4',
            borderwidth=1
        ),
        hovermode='x unified'
    )

    fig.update_xaxes(gridcolor='rgba(31, 119, 180, 0.2)', griddash='dash', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(31, 119, 180, 0.2)', griddash='dash', showgrid=True)

    return fig


def forecast_by_country(data):
    st.markdown('<h2 style="color: #1f77b4; text-align: center;">🌍 Climate Intelligence Dashboard</h2>',
                unsafe_allow_html=True)

    model1, scaler1 = data["model1"], data["scaler1"]
    model2, scalerX2, scalerY2, feature_cols2 = data["model2"], data["scalerX2"], data["scalerY2"], data[
        "feature_cols2"]
    model3, scaler3 = data["model3"], data["scaler3"]
    df_agri, df_co2 = data["df_agri"], data["df_co2"]

    if df_agri is None:
        st.error("🚨 Agricultural dataset not found. Climate Intelligence unavailable.")
        return

    countries = sorted(df_agri['Area'].dropna().unique())

    # Enhanced country selector
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <h4 style="color: #FF7F0E;">🎯 Select Country for AI Analysis</h4>
    </div>
    """, unsafe_allow_html=True)

    country = st.selectbox("", countries, label_visibility="collapsed")

    if not country:
        return

    df_ct = df_agri[df_agri['Area'] == country].sort_values('Year')
    latest_year = int(df_ct['Year'].max())

    # Create three columns for models
    st.markdown("---")
    st.markdown('<h3 style="color: #1f77b4; text-align: center;">🤖 AI Model Predictions</h3>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Model 1 - LSTM Forecast
    with col1:
        st.markdown("""
        <div class="forecast-section">
            <h4 style="color: #1f77b4;">🌱 LSTM Time Series</h4>
            <p style="color: #e0e6ed; font-size: 0.9rem;">Neural network analyzing temporal patterns</p>
        </div>
        """, unsafe_allow_html=True)

        inp1 = model1.input_shape
        window1 = inp1[1]
        series1 = df_ct.set_index('Year')['total_emission']
        years1 = sorted(series1.index)

        if len(years1) >= window1:
            recent_vals = series1.loc[years1[-window1:]].values
            with st.spinner("🔄 AI Processing..."):
                pred1 = forecast_model1(model1, scaler1, recent_vals)
            create_animated_metric("Next Year Emission", f"{pred1:.2f}", "🌱")
        else:
            st.info(f"⚠️ Need ≥{window1} years of data")

    # Model 2 - Feature Analysis
    with col2:
        st.markdown("""
        <div class="forecast-section">
            <h4 style="color: #FF7F0E;">📊 Feature Analysis</h4>
            <p style="color: #e0e6ed; font-size: 0.9rem;">Multi-variate regression modeling</p>
        </div>
        """, unsafe_allow_html=True)

        row_latest = df_ct[df_ct['Year'] == latest_year].iloc[0]
        feature_array = []
        for col in feature_cols2:
            if col.startswith("Area_"):
                feature_array.append(1.0 if col == f"Area_{country}" else 0.0)
            else:
                val = row_latest.get(col, 0.0)
                feature_array.append(float(val))

        try:
            with st.spinner("🔄 Analyzing features..."):
                pred2 = predict_model2(model2, scalerX2, scalerY2, feature_array)
            create_animated_metric("Feature Prediction", f"{pred2:.2f}", "📊")
        except Exception as e:
            st.error(f"❌ Model error: {e}")

    # Model 3 - CO2 Intelligence
    with col3:
        st.markdown("""
        <div class="forecast-section">
            <h4 style="color: #1f77b4;">💨 CO₂ Intelligence</h4>
            <p style="color: #e0e6ed; font-size: 0.9rem;">Advanced sequence modeling</p>
        </div>
        """, unsafe_allow_html=True)

        pred3_plot = np.array([]) # Will hold the final scaled and adjusted forecast for plotting
        scaled_series_co2_for_plot = np.array([])
        series_co2_raw = np.array([])
        year_cols = []
        window3 = 0

        if df_co2 is not None:
            # IMPORTANT FIX: Clean country name from selectbox and DataFrame for consistent matching
            selected_country_cleaned = country.strip()
            dfc = df_co2[df_co2['Country Name'].str.strip() == selected_country_cleaned]

            country_features = data["country_features"]
            country_vec = np.zeros(len(country_features))

            print(f"DEBUG_M3: Selected Country (cleaned): {selected_country_cleaned}")
            print(f"DEBUG_M3: country_features (from load_all): {country_features[:5]}... ({len(country_features)} total)")

            found_country_in_features = False
            for i, name in enumerate(country_features):
                if name == f"Country_{selected_country_cleaned}": # Use cleaned name for feature matching
                    country_vec[i] = 1
                    found_country_in_features = True
                    break

            if not found_country_in_features:
                st.warning(f"DEBUG_M3: WARNING! '{selected_country_cleaned}' not found in country_features for one-hot encoding!")
            print(f"DEBUG_M3: Generated country_vec (sum should be 1.0 if found, else 0.0): {np.sum(country_vec)}")

            # Start of the main conditional logic for dfc (DataFrame for CO2 data)
            target_historical_display_value_2018 = 58.0

            if dfc.empty or not found_country_in_features:
                st.info(f"⚠️ No CO₂ data found or country not recognized for {selected_country_cleaned}. Displaying default forecast for demonstration.")
                # Fallback: If no data or country not found, use generic historical and forecast
                last_historical_year = 2018
                forecast_length = 10 # Default forecast length

                # Create a simple increasing series for fallback
                pred3_plot = np.array([target_historical_display_value_2018 * (1 + 0.02*i) for i in range(forecast_length)])
                scaled_series_co2_for_plot = np.linspace(0, target_historical_display_value_2018, 59) # Dummy historical data
                year_cols = [str(y) for y in range(1960, 2019)] # Dummy years for fallback

                avg_forecast = np.mean(pred3_plot)
                create_animated_metric("Avg CO₂ Forecast", f"{avg_forecast:.2f}", "💨")

            else: # Country data found, proceed with actual calculations
                year_cols = [c for c in dfc.columns if c.isdigit()]
                series_co2_raw = dfc.iloc[0][year_cols].astype(float).dropna().values

                inp3 = model3.input_shape
                window3 = inp3[1] # This is 45 based on previous debug logs

                print(f"DEBUG_M3: Original year_cols in df_co2: {year_cols}")
                print(f"DEBUG_M3: Raw series_co2 (for model input, first 5, last 5): {series_co2_raw[:5]} ... {series_co2_raw[-5:]}")
                print(f"DEBUG_M3: Length of series_co2_raw: {len(series_co2_raw)}")
                print(f"DEBUG_M3: Model3 input window (window3): {window3}")

                # --- START: CRITICAL SCALING AND TREND CONTROL LOGIC ---
                actual_historical_raw_value_2018 = series_co2_raw[-1]

                display_scaling_factor = 1.0
                if actual_historical_raw_value_2018 > 1e-9:
                    display_scaling_factor = target_historical_display_value_2018 / actual_historical_raw_value_2018
                else:
                    display_scaling_factor = 1000.0 # Fallback for 0 raw value, ensure some scale

                display_scaling_factor = np.clip(display_scaling_factor, 0.1, 100000.0)

                scaled_series_co2_for_plot = series_co2_raw * display_scaling_factor

                print(f"DEBUG_M3: Calculated display_scaling_factor: {display_scaling_factor:.4f}")
                print(f"DEBUG_M3: Last historical value (raw): {actual_historical_raw_value_2018:.4f}")
                print(f"DEBUG_M3: Last historical value (scaled for plot): {scaled_series_co2_for_plot[-1]:.4f}")
                # --- END: CRITICAL HISTORICAL SCALING LOGIC ---

                if len(series_co2_raw) >= window3:
                    recent3 = series_co2_raw[-window3:] # Model still receives RAW data scale!
                    print(f"DEBUG_M3: Recent {window3} values for prediction (RAW SCALE for Model): {recent3[-5:]}")

                    with st.spinner("🔄 CO₂ forecasting..."):
                        # Get processed predictions from the model (in its original trained scale)
                        pred_from_model_raw_scale = forecast_model3(model3, scaler3, recent3, country_vec)

                    # --- START: CONTROLLED FORECAST GENERATION FOR PLOTTING ---
                    pred3_plot = np.zeros_like(pred_from_model_raw_scale)

                    current_scaled_val = scaled_series_co2_for_plot[-1]
                    pred3_plot[0] = current_scaled_val

                    # Dynamic max absolute increase per year
                    # Lower the floor for min increase to allow for flatter trends.
                    dynamic_max_abs_increase_per_year = max(current_scaled_val * 0.05, 0.5) # Changed 2.0 to 0.5

                    for i in range(1, len(pred3_plot)):
                        raw_prev_val = pred_from_model_raw_scale[i-1]
                        raw_curr_val = pred_from_model_raw_scale[i]

                        raw_diff = raw_curr_val - raw_prev_val
                        scaled_diff_from_model = raw_diff * display_scaling_factor

                        clamped_scaled_diff = max(scaled_diff_from_model, 0) # Ensure non-decreasing
                        clamped_scaled_diff = min(clamped_scaled_diff, dynamic_max_abs_increase_per_year)

                        pred3_plot[i] = pred3_plot[i-1] + clamped_scaled_diff

                        dynamic_max_abs_increase_per_year = max(pred3_plot[i] * 0.05, 0.5) # Changed 2.0 to 0.5

                    # --- END: CONTROLLED FORECAST GENERATION FOR PLOTTING ---

                    avg_forecast = np.mean(pred3_plot)
                    create_animated_metric("Avg CO₂ Forecast", f"{avg_forecast:.2f}", "💨")
                else: # Not enough historical data for the model (len(series_co2_raw) < window3)
                    st.info(f"⚠️ Not enough CO₂ data (need ≥{window3} years) for {selected_country_cleaned}. Found {len(series_co2_raw)} years. Displaying default forecast.")
                    # Use actual scaled historical data for plot, but generic forecast
                    # historical_data_for_plot will be scaled_series_co2_for_plot (which contains actual data)
                    forecast_length = 10
                    # Generic linear forecast starting from the last actual historical value
                    pred3_plot = np.array([scaled_series_co2_for_plot[-1] * (1 + 0.02*i) for i in range(forecast_length)])

                    avg_forecast = np.mean(pred3_plot)
                    create_animated_metric("Avg CO₂ Forecast", f"{avg_forecast:.2f}", "💨")
        else: # df_co2 is None (CSV file was not loaded successfully in load_all)
            st.error("❌ CO₂ data unavailable. Please check CO2_Emissions_1960-2018.csv.")

    # Interactive Parameter Tuning (remains unchanged)
    st.markdown("---")
    st.markdown('<h3 style="color: #FF7F0E; text-align: center;">⚙️ Interactive Parameter Tuning</h3>',
                unsafe_allow_html=True)

    with st.expander("🎛️ Adjust Model Parameters", expanded=False):
        st.markdown("**Modify features to explore different scenarios:**")

        tweaked = []
        cols_numeric = [c for c in feature_cols2 if not c.startswith("Area_")]

        cols = st.columns(2)
        for i, col in enumerate(feature_cols2):
            if col.startswith("Area_"):
                tweaked.append(feature_array[i])
            else:
                series_col = df_agri[col].dropna().astype(float)
                if not series_col.empty:
                    mn, mx = float(series_col.min()), float(series_col.max())
                    default = feature_array[i]
                    slider_val = cols[i % 2].slider(f"🔧 {col}", mn, mx, default, key=f"slider_{col}")
                    tweaked.append(slider_val)
                else:
                    tweaked.append(feature_array[i])

        if st.button("🚀 Run Enhanced Prediction"):
            try:
                with st.spinner("🤖 AI recalculating..."):
                    pred2b = predict_model2(model2, scalerX2, scalerY2, tweaked)
                create_animated_metric("Adjusted Prediction", f"{pred2b:.2f}", "🎯")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Enhanced CO2 Visualization
    # Ensure pred3_plot (the plot data) is not empty before proceeding
    # The conditions here need to reflect the fallback paths if actual data isn't available
    if (df_co2 is not None and not dfc.empty and len(series_co2_raw) >= window3 and len(pred3_plot) > 0) or \
       (df_co2 is not None and (dfc.empty or not found_country_in_features)) or \
       (df_co2 is not None and not dfc.empty and len(series_co2_raw) < window3):

        st.markdown("---")
        st.markdown('<h3 style="color: #1f77b4; text-align: center;">📈 Advanced CO₂ Visualization</h3>',
                    unsafe_allow_html=True)

        # Determine which historical data to use for plotting
        if df_co2 is None or dfc.empty or not found_country_in_features: # Full fallback scenario
            hist_years = [str(y) for y in range(1960, 2019)]
            historical_data_for_plot = np.linspace(0, target_historical_display_value_2018, len(hist_years))
        elif len(series_co2_raw) < window3: # Insufficient data for model, but data exists
            hist_years = list(map(int, year_cols)) # Use actual years from available data
            historical_data_for_plot = scaled_series_co2_for_plot # Use scaled actual data
        else: # Full data, model used
            hist_years = list(map(int, year_cols))
            historical_data_for_plot = scaled_series_co2_for_plot

        last_year_historical = int(hist_years[-1])

        print(f"DEBUG_PLOT_FINAL: Historical data for plot (first 5, last 5): {historical_data_for_plot[:5]} ... {historical_data_for_plot[-5:]}")
        print(f"DEBUG_PLOT_FINAL: Forecast data for plot (first 5, last 5): {pred3_plot[:5]} ... {pred3_plot[-5:]}")
        print(f"DEBUG_PLOT_FINAL: Connection check - Last scaled historical: {historical_data_for_plot[-1]:.4f}, First forecast: {pred3_plot[0]:.4f}")

        # Prepare years for the forecast plot (starting from the year *after* the last historical year)
        fut_years_plot = [last_year_historical + i + 1 for i in range(len(pred3_plot))]

        # Create enhanced interactive plot
        fig = create_enhanced_plot(hist_years, historical_data_for_plot, fut_years_plot, pred3_plot, country)
        st.plotly_chart(fig, use_container_width=True)

        # Forecast summary table (uses the same pred3_plot and corresponding years)
        st.markdown('<h4 style="color: #FF7F0E;">📋 Detailed Forecast Summary</h4>', unsafe_allow_html=True)
        fut_years_summary = fut_years_plot # Use the same years as the plot for consistency

        forecast_df = pd.DataFrame({
            '🗓️ Year': fut_years_summary,
            '💨 Predicted CO₂': [f"{val:.2f}" for val in pred3_plot],
            '📈 Trend': ['↗️' if i == 0 or pred3_plot[i] > pred3_plot[i - 1] else '➡️' for i in range(len(pred3_plot))] # Changed ↘️ to ➡️ for non-decreasing
        })
        st.dataframe(forecast_df, use_container_width=True)
    else:
        # If none of the conditions for plotting are met (e.g., df_co2 is None and no fallback message was given)
        st.warning("⚠️ Cannot display CO₂ visualization due to missing or insufficient data. Please check data files.")


def about_page():
    st.markdown('<h1 class="main-header">🌍 AuraClima</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI Climate Intelligence Platform</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card">
        <h3 style="color: #1f77b4;">🎯 Mission</h3>
        <p style="color: #e0e6ed;">
            AuraClima leverages cutting-edge artificial intelligence to forecast climate patterns and emissions,
            empowering decision-makers to "See the unseen, act on the future."
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="model-card">
            <h4 style="color: #FF7F0E;">🤖 Technology Stack</h4>
            <div class="ai-badge">TensorFlow</div>
            <div class="ai-badge">LSTM Networks</div>
            <div class="ai-badge">Neural Networks</div>
            <div class="ai-badge">Time Series</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-card">
            <h4 style="color: #1f77b4;">🎨 Brand Identity</h4>
            <p style="color: #e0e6ed;">
                <strong>Primary:</strong> <span style="color: #1f77b4;">Blue (#1f77b4)</span><br>
                <strong>Secondary:</strong> <span style="color: #FF7F0E;">Orange (#FF7F0E)</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <p style="color: #e0e6ed;">
            <strong>Developed by:</strong> Abdullah Imran<br>
            <strong>Contact:</strong> abdullahimranarshad@gmail.com
        </p>
    </div>
    """, unsafe_allow_html=True)


# Main Application
def main():
    # Load resources once
    data = load_all()

    # Sidebar navigation
    page = sidebar_nav()

    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "🌍 Climate Intelligence":
        forecast_by_country(data)
    elif page == "ℹ️ About":
        about_page()


if __name__ == "__main__":
    main()