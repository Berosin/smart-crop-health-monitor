"""Analysis History page — browse past analyses (dummy rows, no DB yet)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ui import (
    page_header,
    callout,
    footer,
    get_dummy_history,
)


def render() -> None:
    page_header(
        "🗂️",
        "Analysis History",
        "Review previously saved crop analyses.",
    )

    history = get_dummy_history()
    df = pd.DataFrame(history)

    st.markdown("#### Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        crop_filter = st.multiselect("Crop", df["crop"].unique(), default=list(df["crop"].unique()))
    with f2:
        status_filter = st.multiselect("Status", df["status"].unique(), default=list(df["status"].unique()))
    with f3:
        min_score = st.slider("Min health score", 0, 100, 0, 5)

    mask = (df["crop"].isin(crop_filter)) & (df["status"].isin(status_filter)) & (df["health_score"] >= min_score)
    view = df[mask].sort_values("date", ascending=False)

    st.markdown(f"#### {len(view)} analyses")
    st.dataframe(
        view,
        use_container_width=True,
        column_config={
            "id": "ID",
            "date": "Date",
            "crop": "Crop",
            "status": st.column_config.TextColumn("Status"),
            "health_score": st.column_config.ProgressColumn("Health score", min_value=0, max_value=100, format="%d"),
            "temperature": "Temp (°C)",
            "humidity": "Humidity (%)",
            "soil_moisture": "Soil moist. (%)",
            "rainfall": "Rainfall (mm)",
        },
        hide_index=True,
    )

    # Expandable detail cards
    with st.expander("Detail view"):
        idx = st.selectbox("Select an analysis to inspect", view["id"].tolist())
        row = view[view["id"] == idx].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Crop", row["crop"])
        c2.metric("Status", row["status"])
        c3.metric("Health score", row["health_score"])
        c4.metric("Date", row["date"])
        st.caption("Field image and full recommendation would be shown here once stored.")

    callout("🗂️ Showing dummy history rows. Real records will appear once the "
            "SQLite layer is implemented.")
    footer()
