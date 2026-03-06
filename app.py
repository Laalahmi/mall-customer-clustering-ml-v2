import pandas as pd
import streamlit as st
from pathlib import Path
from src.artifacts import load_model_bundle
from src.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_bundle():
    return load_model_bundle()


def predict_cluster(model, scaler, features, input_data: pd.DataFrame):
    """
    Predict cluster for a new customer input.
    """
    scaled_input = scaler.transform(input_data[features])
    cluster = model.predict(scaled_input)[0]
    return int(cluster)


def get_logo_path():
    return Path("assets") / "algonquin_logo.png"


def load_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .hero-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 2rem;
            border-radius: 22px;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #e2e8f0;
            margin-bottom: 0.5rem;
        }

        .section-card {
            background: white;
            padding: 1.25rem 1.25rem 1rem 1.25rem;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }

        .mini-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
            text-align: center;
        }

        .mini-card h4 {
            margin: 0;
            color: #475569;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .mini-card p {
            margin: 0.35rem 0 0 0;
            font-size: 1.4rem;
            font-weight: 800;
            color: #0f172a;
        }

        .footer-card {
            background: #0f172a;
            color: white;
            padding: 1rem 1.25rem;
            border-radius: 18px;
            margin-top: 1.5rem;
            text-align: center;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }

        .credit-line {
            font-size: 0.95rem;
            color: #cbd5e1;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #e2e8f0;
            border-radius: 12px;
            padding: 10px 18px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background-color: #1d4ed8 !important;
            color: white !important;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #e5e7eb;
            padding: 12px;
            border-radius: 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }

        .cluster-tag {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: #dbeafe;
            color: #1d4ed8;
            font-weight: 700;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def show_header(best_k):
    logo_path = get_logo_path()

    col1, col2 = st.columns([1, 5])

    with col1:
        if logo_path.exists():
            st.image(str(logo_path), width=130)

    with col2:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-title">Mall Customer Segmentation Dashboard</div>
                <div class="hero-subtitle">
                    Unsupervised learning project using KMeans clustering to segment customers
                    based on Age, Annual Income, and Spending Score.
                </div>
                <div class="cluster-tag">Best Number of Clusters: {best_k}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def interpret_cluster(profile_row: pd.Series) -> str:
    income = profile_row["Annual_Income"]
    spending = profile_row["Spending_Score"]
    age = profile_row["Age"]

    if income >= 75 and spending >= 70:
        return "High-income, high-spending customers. Likely premium or highly engaged customers."
    elif income >= 75 and spending < 40:
        return "High-income but low-spending customers. May represent under-engaged or cautious customers."
    elif income < 40 and spending >= 60:
        return "Lower-income but high-spending customers. Could reflect young, active, impulse-oriented shoppers."
    elif income < 40 and spending < 40:
        return "Lower-income, lower-spending customers. May represent more conservative spending behavior."
    elif age < 30:
        return "Younger customer segment with moderate purchasing behavior."
    else:
        return "Balanced customer segment with moderate income and spending patterns."


def main():
    load_custom_css()
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use this dashboard to explore the clustering model and predict customer segments.")

    try:
        bundle = load_bundle()
        model = bundle["model"]
        scaler = bundle["scaler"]
        features = bundle["features"]
        cluster_summary = bundle["cluster_summary"]
        best_k = bundle["best_k"]
        search_results = bundle.get("search_results", [])

        show_header(best_k)

        st.sidebar.success("Model bundle loaded successfully")
        st.sidebar.metric("Selected K", best_k)
        st.sidebar.metric("Number of Features", len(features))
        st.sidebar.metric("Discovered Segments", len(cluster_summary))

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Best K", best_k)
        with metric_col2:
            best_silhouette = max([r["silhouette"] for r in search_results]) if search_results else 0
            st.metric("Best Silhouette", f"{best_silhouette:.4f}")
        with metric_col3:
            st.metric("Segments Created", len(cluster_summary))

        tab1, tab2, tab3 = st.tabs(["Overview", "Cluster Insights", "Predict Segment"])

        with tab1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Project Overview")
            st.write(
                "This application uses KMeans clustering to group mall customers into meaningful segments. "
                "The model was trained using three numeric features: Age, Annual Income, and Spending Score."
            )
            st.write(
                "The goal is to support business understanding by identifying groups of customers with similar "
                "behavioral and financial characteristics."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if search_results:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("K Search Results")
                results_df = pd.DataFrame(search_results)

                st.dataframe(results_df, use_container_width=True)
                st.line_chart(results_df.set_index("k")[["inertia", "silhouette"]])
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Cluster Profiles")
            st.dataframe(cluster_summary, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Segment Interpretation")

            for cluster_id in cluster_summary.index:
                row = cluster_summary.loc[cluster_id]
                interpretation = interpret_cluster(row)

                st.markdown(
                    f"""
                    <div class="mini-card" style="margin-bottom: 12px; text-align:left;">
                        <h4>Cluster {cluster_id}</h4>
                        <p style="font-size:1rem; font-weight:500; color:#334155; margin-top:8px;">
                            Average Age: <b>{row['Age']:.2f}</b><br>
                            Average Annual Income: <b>{row['Annual_Income']:.2f}</b><br>
                            Average Spending Score: <b>{row['Spending_Score']:.2f}</b><br><br>
                            {interpretation}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Predict Customer Segment")
            st.write("Enter customer details below to estimate the cluster to which the customer belongs.")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)

            with col2:
                annual_income = st.number_input("Annual Income", min_value=0, max_value=200, value=60, step=1)

            with col3:
                spending_score = st.number_input("Spending Score", min_value=1, max_value=100, value=50, step=1)

            if st.button("Predict Cluster", use_container_width=True):
                input_df = pd.DataFrame([{
                    "Age": age,
                    "Annual_Income": annual_income,
                    "Spending_Score": spending_score
                }])

                cluster = predict_cluster(model, scaler, features, input_df)

                st.success(f"Predicted Cluster: {cluster}")

                if cluster in cluster_summary.index:
                    st.markdown("### Predicted Cluster Profile")
                    st.dataframe(cluster_summary.loc[[cluster]], use_container_width=True)

                    interpretation = interpret_cluster(cluster_summary.loc[cluster])
                    st.info(interpretation)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="footer-card">
                <div><b>Algonquin College — CST2216 Modularizing and Deploying ML Code</b></div>
                <div class="credit-line">Created by Mohammed Laalahmi</div>
                <div class="credit-line">Professor: Dr. Umer Altaf</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as exc:
        logger.exception("Application error.")
        st.error(f"An error occurred while loading the application: {exc}")


if __name__ == "__main__":
    main()