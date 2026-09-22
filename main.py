import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib

from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GeoTherm-AI",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("persistent_sources.csv")

    df["static_ratio"] = (
        df["static_detections"] /
        df["detections"]
    )

    df["vegetation_ratio"] = (
        df["vegetation_detections"] /
        df["detections"]
    )

    def classify_source(row):

        if row["static_ratio"] >= 0.90:
            return "Static-dominant"

        elif row["vegetation_ratio"] >= 0.50:
            return "Vegetation-dominant"

        else:
            return "Mixed"

    df["source_profile"] = df.apply(
        classify_source,
        axis=1
    )

    return df


@st.cache_resource
def load_model():

    model = joblib.load(
        "geotherm_model.pkl"
    )

    features = joblib.load(
        "model_features.pkl"
    )

    return model, features


df = load_data()
model, model_features = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("🔥 GeoTherm-AI")

st.markdown(
    """
    ### AI-Based Industrial Thermal Anomaly Detection & Monitoring

    **VIIRS thermal anomaly → persistence analysis → ML classification**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Detection Filters")


min_active_days = st.sidebar.slider(
    "Minimum Active Days",
    min_value=1,
    max_value=int(df["active_days"].max()),
    value=10
)


min_detections = st.sidebar.slider(
    "Minimum Detections",
    min_value=1,
    max_value=int(df["detections"].max()),
    value=10
)


source_profiles = st.sidebar.multiselect(
    "Source Profile",
    options=[
        "Static-dominant",
        "Mixed",
        "Vegetation-dominant"
    ],
    default=[
        "Static-dominant",
        "Mixed",
        "Vegetation-dominant"
    ]
)


max_points = st.sidebar.slider(
    "Maximum Map Sources",
    min_value=50,
    max_value=700,
    value=300,
    step=50
)


# ============================================================
# FILTER
# ============================================================

filtered_df = df[
    (df["active_days"] >= min_active_days)
    & (df["detections"] >= min_detections)
    & (df["source_profile"].isin(source_profiles))
].copy()


filtered_df = filtered_df.sort_values(
    ["active_days", "detections"],
    ascending=False
)


# ============================================================
# KPI
# ============================================================

st.subheader("📊 Detection Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Persistent Sources",
        f"{len(filtered_df):,}"
    )


with col2:
    st.metric(
        "Total Detections",
        f"{filtered_df['detections'].sum():,}"
    )


with col3:

    max_days = (
        int(filtered_df["active_days"].max())
        if len(filtered_df) > 0
        else 0
    )

    st.metric(
        "Maximum Persistence",
        f"{max_days} days"
    )


with col4:

    max_frp = (
        filtered_df["max_frp"].max()
        if len(filtered_df) > 0
        else 0
    )

    st.metric(
        "Maximum FRP",
        f"{max_frp:.2f} MW"
    )


# ============================================================
# SOURCE PROFILE
# ============================================================

st.divider()

st.subheader("🔥 Source Profile")

profile_counts = (
    filtered_df["source_profile"]
    .value_counts()
)


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Static-dominant",
        f"{profile_counts.get('Static-dominant', 0):,}"
    )


with col2:
    st.metric(
        "Mixed",
        f"{profile_counts.get('Mixed', 0):,}"
    )


with col3:
    st.metric(
        "Vegetation-dominant",
        f"{profile_counts.get('Vegetation-dominant', 0):,}"
    )


# ============================================================
# MAP
# ============================================================

st.divider()

st.subheader("🌍 Persistent Thermal Sources")

st.caption(
    "Blue = static-dominant | "
    "Orange = mixed | "
    "Red = vegetation-dominant"
)


map_df = filtered_df.head(max_points)


m = folium.Map(
    location=[22.5, 80.0],
    zoom_start=5,
    tiles="OpenStreetMap"
)


for _, row in map_df.iterrows():

    if row["source_profile"] == "Static-dominant":
        marker_color = "blue"

    elif row["source_profile"] == "Mixed":
        marker_color = "orange"

    else:
        marker_color = "red"


    radius = max(
        4,
        min(
            15,
            row["active_days"] / 15
        )
    )


    popup_html = f"""
    <div style="width:300px">

        <h3>🔥 GeoTherm-AI Thermal Source</h3>

        <b>Source Profile:</b>
        {row["source_profile"]}

        <hr>

        <b>Latitude:</b>
        {row["grid_lat"]:.2f}<br>

        <b>Longitude:</b>
        {row["grid_lon"]:.2f}<br><br>

        <b>Active Days:</b>
        {int(row["active_days"])}<br>

        <b>Detections:</b>
        {int(row["detections"]):,}<br>

        <b>Mean FRP:</b>
        {row["mean_frp"]:.2f} MW<br>

        <b>Maximum FRP:</b>
        {row["max_frp"]:.2f} MW<br>

        <b>Mean Brightness:</b>
        {row["mean_bright_ti4"]:.2f} K<br>

    </div>
    """


    folium.CircleMarker(
        location=[
            row["grid_lat"],
            row["grid_lon"]
        ],
        radius=radius,
        color=marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.65,
        weight=1,
        popup=folium.Popup(
            popup_html,
            max_width=350
        ),
        tooltip=(
            f"{row['source_profile']} | "
            f"{int(row['active_days'])} active days"
        )
    ).add_to(m)


st_folium(
    m,
    width=None,
    height=600
)


# ============================================================
# ML MODEL
# ============================================================

st.divider()

st.subheader("🤖 GeoTherm-AI Classification")

st.caption(
    "LightGBM baseline trained on VIIRS source-type data."
)


# ------------------------------------------------------------
# Create model input from persistent source
# ------------------------------------------------------------

prediction_df = filtered_df.head(max_points).copy()


# The model expects these features.
# For aggregated persistent sources, we construct
# the required feature vector from available statistics.

if len(prediction_df) > 0:

    prediction_features = pd.DataFrame(index=prediction_df.index)

    prediction_features["latitude"] = (
        prediction_df["grid_lat"]
    )

    prediction_features["longitude"] = (
        prediction_df["grid_lon"]
    )

    prediction_features["bright_ti4"] = (
        prediction_df["mean_bright_ti4"]
    )

    prediction_features["bright_ti5"] = (
        prediction_df["mean_bright_ti5"]
        if "mean_bright_ti5" in prediction_df.columns
        else prediction_df["mean_bright_ti4"]
    )

    prediction_features["scan"] = 0
    prediction_features["track"] = 0

    prediction_features["acq_minutes"] = 720

    prediction_features["confidence_score"] = 1

    prediction_features["is_day"] = 1

    prediction_features["frp"] = (
        prediction_df["mean_frp"]
    )

    prediction_features["brightness_difference"] = (
        prediction_features["bright_ti4"]
        -
        prediction_features["bright_ti5"]
    )

    prediction_features["detections"] = (
        prediction_df["detections"]
    )

    prediction_features["active_days"] = (
        prediction_df["active_days"]
    )

    prediction_features["mean_frp"] = (
        prediction_df["mean_frp"]
    )

    prediction_features["max_frp"] = (
        prediction_df["max_frp"]
    )

    prediction_features["mean_bright_ti4"] = (
        prediction_df["mean_bright_ti4"]
    )

    prediction_features["max_bright_ti4"] = (
        prediction_df["max_bright_ti4"]
    )

    prediction_features["mean_bright_ti5"] = (
        prediction_df["mean_bright_ti5"]
        if "mean_bright_ti5" in prediction_df.columns
        else prediction_df["mean_bright_ti4"]
    )

    prediction_features["max_bright_ti5"] = (
        prediction_df["max_bright_ti5"]
        if "max_bright_ti5" in prediction_df.columns
        else prediction_df["max_bright_ti4"]
    )


    # Make sure columns are in exactly
    # the same order as training

    prediction_features = prediction_features[
        model_features
    ]


    predictions = model.predict(
        prediction_features
    )

    probabilities = model.predict_proba(
        prediction_features
    )


    prediction_df["ML Prediction"] = np.where(
        predictions == 0,
        "Vegetation Fire",
        "Static Land Source"
    )


    prediction_df["ML Confidence"] = (
        probabilities.max(axis=1) * 100
    ).round(1)


    # --------------------------------------------------------
    # ML summary
    # --------------------------------------------------------

    ml_col1, ml_col2 = st.columns(2)


    with ml_col1:

        static_predictions = (
            prediction_df["ML Prediction"]
            == "Static Land Source"
        ).sum()

        st.metric(
            "Predicted Static Sources",
            f"{static_predictions:,}"
        )


    with ml_col2:

        vegetation_predictions = (
            prediction_df["ML Prediction"]
            == "Vegetation Fire"
        ).sum()

        st.metric(
            "Predicted Vegetation Fires",
            f"{vegetation_predictions:,}"
        )


    # --------------------------------------------------------
    # ML table
    # --------------------------------------------------------

    st.dataframe(
        prediction_df[
            [
                "grid_lat",
                "grid_lon",
                "active_days",
                "detections",
                "ML Prediction",
                "ML Confidence"
            ]
        ].rename(
            columns={
                "grid_lat": "Latitude",
                "grid_lon": "Longitude",
                "active_days": "Active Days",
                "detections": "Detections"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FULL SOURCE TABLE
# ============================================================

st.divider()

st.subheader("📋 Persistent Source Data")

display_columns = [
    "grid_lat",
    "grid_lon",
    "source_profile",
    "detections",
    "active_days",
    "static_ratio",
    "mean_frp",
    "max_frp",
    "static_detections",
    "vegetation_detections"
]


display_df = filtered_df[
    display_columns
].copy()


display_df["static_ratio"] = (
    display_df["static_ratio"] * 100
).round(1)


display_df = display_df.rename(
    columns={
        "grid_lat": "Latitude",
        "grid_lon": "Longitude",
        "source_profile": "Source Profile",
        "detections": "Detections",
        "active_days": "Active Days",
        "static_ratio": "Static %",
        "mean_frp": "Mean FRP (MW)",
        "max_frp": "Max FRP (MW)",
        "static_detections": "Static Detections",
        "vegetation_detections": "Vegetation Detections"
    }
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)