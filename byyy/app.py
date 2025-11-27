import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from streamlit_local_storage import LocalStorage

st.set_page_config(page_title="Netflix Recommender Dashboard", layout="wide")

localS = LocalStorage()

st.title("Netflix Recommendation System – Data Cleaner & Dashboard")

st.markdown(
    "Upload a Netflix-style viewing dataset (CSV). "
    "The app will clean it, compute KPIs, and show visualizations."
)

# Try to restore last-used settings from local storage
if "last_columns" not in st.session_state:
    st.session_state["last_columns"] = None

local_cols = localS.getItem("netflix_last_columns", key="get_cols")
if local_cols and st.session_state["last_columns"] is None:
    st.session_state["last_columns"] = local_cols[0]

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standard column name normalization
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop duplicates
    df = df.drop_duplicates()

    # Basic missing handling
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Drop rows where critical identifiers are missing (user/movie)
    for key_col in ["user_id", "movie_id"]:
        if key_col in df.columns:
            df = df[df[key_col].notna()]

    # Try to parse timestamp column if present
    for tcol in ["timestamp", "date", "watch_date"]:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
            break

    return df

def compute_kpis(df: pd.DataFrame):
    total_users = df["user_id"].nunique() if "user_id" in df.columns else np.nan
    total_titles = df["movie_id"].nunique() if "movie_id" in df.columns else np.nan
    avg_rating = df["rating"].mean() if "rating" in df.columns else np.nan
    total_watch_minutes = (
        df["watch_duration_min"].sum() if "watch_duration_min" in df.columns else np.nan
    )
    return total_users, total_titles, avg_rating, total_watch_minutes

def show_kpi_cards(total_users, total_titles, avg_rating, total_watch_minutes):
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Unique Users", f"{int(total_users):,}" if not np.isnan(total_users) else "N/A")
    kpi2.metric("Unique Titles", f"{int(total_titles):,}" if not np.isnan(total_titles) else "N/A")
    kpi3.metric("Average Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
    kpi4.metric(
        "Total Watch Time (hrs)",
        f"{total_watch_minutes/60:.1f}" if not np.isnan(total_watch_minutes) else "N/A",
    )

def show_visualizations(df: pd.DataFrame):
    st.subheader("Visualizations")

    # Genre distribution
    genre_col = None
    for gcol in ["genre_primary", "genre", "main_genre"]:
        if gcol in df.columns:
            genre_col = gcol
            break

    # Rating distribution
    if "rating" in df.columns:
        fig_rating = px.histogram(df, x="rating", nbins=10, title="Rating Distribution")
        st.plotly_chart(fig_rating, use_container_width=True)

    # Watch duration by genre
    if genre_col and "watch_duration_min" in df.columns:
        fig_genre = px.box(
            df,
            x=genre_col,
            y="watch_duration_min",
            title="Watch Duration by Genre",
        )
        st.plotly_chart(fig_genre, use_container_width=True)

    # Top titles by watch time
    if "movie_title" in df.columns and "watch_duration_min" in df.columns:
        top_titles = (
            df.groupby("movie_title")["watch_duration_min"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig_titles = px.bar(
            top_titles,
            x="watch_duration_min",
            y="movie_title",
            orientation="h",
            title="Top 15 Titles by Total Watch Time",
        )
        fig_titles.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_titles, use_container_width=True)

if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.read()
        df_raw = pd.read_csv(io.BytesIO(raw_bytes))
        df = clean_data(df_raw)

        # Persist basic metadata to local storage (column names)
        cols_info = list(df.columns)
        localS.setItem("netflix_last_columns", cols_info, key="set_cols")
        st.session_state["last_columns"] = cols_info

        st.success("File uploaded and cleaned successfully.")
        st.write("Preview of cleaned data:")
        st.dataframe(df.head(20))

        # KPIs
        total_users, total_titles, avg_rating, total_watch_minutes = compute_kpis(df)
        show_kpi_cards(total_users, total_titles, avg_rating, total_watch_minutes)

        # Sidebar filters (example)
        st.sidebar.header("Filters")
        if "country" in df.columns:
            countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
            country_filter = st.sidebar.selectbox("Country", countries)
            if country_filter != "All":
                df = df[df["country"] == country_filter]

        if "genre_primary" in df.columns:
            genres = ["All"] + sorted(df["genre_primary"].dropna().unique().tolist())
            genre_filter = st.sidebar.selectbox("Primary Genre", genres)
            if genre_filter != "All":
                df = df[df["genre_primary"] == genre_filter]

        # Visualizations
        show_visualizations(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin.")
    if st.session_state["last_columns"]:
        st.caption(
            "Last used dataset columns (from browser local storage): "
            + ", ".join(st.session_state["last_columns"])
        )
