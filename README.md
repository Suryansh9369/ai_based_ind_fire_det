# GeoTherm-AI

### AI-Enabled Geospatial System for Industrial Thermal Anomaly Detection, Classification & Monitoring

GeoTherm-AI is a prototype geospatial AI system designed to detect, contextualize, classify, and monitor thermal anomalies using satellite and geospatial data.

The system combines satellite thermal observations, satellite imagery, land-cover information, industrial infrastructure data, and historical observations to determine the likely source of detected thermal anomalies.

---

## 🚨 Problem

Satellite-based systems can detect thermal anomalies, but a detected hotspot does not always indicate the same type of event.

A thermal anomaly may correspond to:

- Industrial Fire
- Gas Flare
- Agricultural Burning
- Wildfire
- Mining / Industrial Activity
- Other / Unknown thermal sources

The prototype addresses this problem by adding spatial, temporal, land-cover, and industrial-facility context to satellite thermal observations.

---

# 🎯 Objectives

GeoTherm-AI aims to:

1. Detect satellite-based thermal anomalies.
2. Associate anomalies with nearby industrial facilities.
3. Identify surrounding land-cover characteristics.
4. Analyse repeated thermal observations over time.
5. Classify the likely source of a thermal anomaly.
6. Store historical thermal events.
7. Visualize classified events through a GIS dashboard.
8. Provide confidence and persistence information to users.

---

# 🛰️ Data Sources

The prototype architecture supports multiple data sources.

### NASA FIRMS / VIIRS

Provides satellite thermal anomaly observations such as:

- Latitude / Longitude
- Detection time
- Brightness temperature
- Fire Radiative Power (FRP)
- Confidence
- Thermal anomaly information

### Sentinel-2

Used for:

- Multispectral satellite imagery
- Local area context
- Land-surface characteristics
- Image-based feature extraction

### Land-Cover Data

Used to determine whether an anomaly is located in or near:

- Industrial areas
- Agricultural areas
- Forests
- Urban areas
- Water
- Other land classes

### Industrial / Geospatial Data

Contains information about:

- Industrial facilities
- Refineries
- Power plants
- Mining sites
- Petrochemical facilities
- Other infrastructure

---

# 🧠 System Architecture

```text
                    ┌──────────────────────────┐
                    │      DATA SOURCES        │
                    │                          │
                    │ FIRMS / VIIRS            │
                    │ Sentinel-2               │
                    │ Land-Cover Data          │
                    │ Industrial Databases     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ DATA INGESTION &          │
                    │ PREPROCESSING             │
                    │                          │
                    │ Cleaning                 │
                    │ Coordinate Normalization │
                    │ Timestamp Alignment      │
                    │ Image Filtering           │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ GEOSPATIAL DATA FUSION   │
                    │                          │
                    │ Facility Matching        │
                    │ Land-Cover Mapping       │
                    │ Spatial Intersection     │
                    │ Image Region Extraction  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ FEATURE EXTRACTION       │
                    │                          │
                    │ Thermal Features         │
                    │ Spatial Features         │
                    │ Image Features           │
                    │ Temporal Features        │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
      ┌─────────────────────┐         ┌─────────────────────┐
      │ TEMPORAL /          │         │ AI CLASSIFICATION   │
      │ PERSISTENCE         │────────▶│ ENGINE              │
      │ ANALYSIS            │         │                     │
      │                     │         │ LightGBM / XGBoost  │
      │ Repeated Events     │         │ CNN / ViT            │
      │ Persistence Score   │         │ Feature Fusion       │
      └─────────────────────┘         └──────────┬──────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │ EVENT CLASSIFICATION   │
                                    │                        │
                                    │ Industrial Fire        │
                                    │ Gas Flare              │
                                    │ Agricultural Burning   │
                                    │ Wildfire               │
                                    │ Mining / Industrial    │
                                    │ Other / Unknown        │
                                    └────────────┬───────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │ PostgreSQL + PostGIS   │
                                    │                        │
                                    │ Thermal Events         │
                                    │ Classification Results │
                                    │ Facility Relationships │
                                    │ Historical Events      │
                                    │ Persistence Records    │
                                    └────────────┬───────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │       FastAPI          │
                                    │     API Layer          │
                                    └────────────┬───────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │   GIS WEB DASHBOARD    │
                                    │                        │
                                    │ Interactive Map        │
                                    │ Event Classification   │
                                    │ Facility Information   │
                                    │ Persistence Timeline   │
                                    │ Confidence             │
                                    └────────────┬───────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │       END USERS         │
                                    │                        │
                                    │ Disaster Management    │
                                    │ Industrial Safety      │
                                    │ Environmental Agencies │
                                    │ GIS Analysts           │
                                    └────────────────────────┘