import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Netflix Recommendation Dashboard", layout="wide")

# ---------- LOCAL STORAGE ----------
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

st.title("🎬 Netflix Recommendation System – Dashboard")

st.write("Upload your Netflix dataset (.csv) to begin:")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ---------- CLEANING FUNCTION ----------
def clean_data(df):
    df = df.copy()

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Handle missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

# ---------- PROCESS DATA ----------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Uploaded Data")
    st.dataframe(df.head())

    cleaned = clean_data(df)
    st.session_state.clean_df = cleaned

    st.success("Data cleaned successfully!")

# ---------- SHOW DASHBOARD ----------
if st.session_state.clean_df is not None:
    df = st.session_state.clean_df

    st.header("📊 Netflix Dashboard")

    # ---- KPI Cards ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Titles", len(df))

    with col2:
        st.metric("Avg Rating", round(df["rating"].mean(), 2))

    with col3:
        st.metric("Avg Duration (mins)", round(df["duration"].mean(), 1))

    with col4:
        st.metric("Most Common Genre", df["genre"].mode()[0])

    # ---- Charts Section ----
    st.subheader("Genre Distribution")
    genre_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="genre:N",
            y="count()",
            color="genre:N",
            tooltip=["genre", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(genre_chart, use_container_width=True)

    st.subheader("Rating Distribution")
    rating_chart = (
        alt.Chart(df)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("rating:Q", bin=True),
            y="count()",
            tooltip=["count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(rating_chart, use_container_width=True)

    st.subheader("Release Year Trend")
    year_chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="release_year:O",
            y="count()",
            tooltip=["release_year", "count()"],
        )
        .properties(height=300)
    )
    st.altair_chart(year_chart, use_container_width=True)

else:
    st.info("Please upload a dataset to view dashboard.")
