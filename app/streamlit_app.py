"""Fraud Detection & Risk Analytics dashboard.

Run from the project root:   streamlit run app/streamlit_app.py

Reads only small derived artefacts (reports/, models/); the raw dataset is not needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.explainability import contribution_table, explain_rows, get_explainer, plot_waterfall  # noqa: E402
from src.features import features_from_card_summary  # noqa: E402
from src.risk import assign_tier, load_thresholds  # noqa: E402

st.set_page_config(page_title="Fraud Risk Analytics", page_icon="🛡️", layout="wide")

SQL = config.REPORTS_DIR / "sql"
FRAUD, LEGIT, MUTED = config.COLOR_FRAUD, config.COLOR_LEGIT, config.COLOR_TEXT_MUTED
TIER_STYLE = {"Low": ("✅", "#008300"), "Medium": ("⚠️", "#c98500"), "High": ("⛔", "#d03b3b")}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ------------------------------------------------------------------ loaders
@st.cache_data
def read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource
def load_model():
    path = config.MODELS_DIR / "xgb_pipeline.joblib"
    if not path.exists():
        return None, None
    pipe = joblib.load(path)
    return pipe, get_explainer(pipe)


def missing(what: str, command: str) -> None:
    st.warning(f"{what} not found. Generate it with:\n\n```\n{command}\n```")


def image(name: str, caption: str | None = None) -> None:
    path = config.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


# ------------------------------------------------------------------ pages
def page_overview() -> None:
    st.title("Financial Transaction Fraud Detection")
    st.caption("Synthetic credit-card transactions (Hugging Face: pointe77/credit-card-transaction). "
               "Train: Jan 2019 – Jun 2020 · Test: Jun 2020 – Dec 2020 (out-of-time).")
    overview = read_csv(SQL / "overview_by_split.csv")
    amounts = read_csv(SQL / "amount_by_fraud_status.csv")
    summary = read_json(config.REPORTS_DIR / "final_model_summary.json")
    if overview is None or amounts is None:
        return missing("SQL outputs", "python -m src.run_sql")

    total = int(overview["n_transactions"].sum())
    n_fraud = int(overview["n_fraud"].sum())
    avg_amt = (amounts["avg_amount"] * amounts["n_transactions"]).sum() / amounts["n_transactions"].sum()
    fraud_avg = float(amounts.loc[amounts["status"] == "fraud", "avg_amount"].iloc[0])
    c = st.columns(4)
    c[0].metric("Total transactions", f"{total:,}")
    c[1].metric("Fraud rate", f"{n_fraud / total:.3%}", help=f"{n_fraud:,} fraudulent transactions")
    c[2].metric("Average amount", f"${avg_amt:,.2f}")
    c[3].metric("Average fraud amount", f"${fraud_avg:,.2f}")

    st.subheader("Model performance on the held-out test period (XGBoost)")
    if summary is None:
        return missing("Model summary", "python -m src.train")
    m = summary["test_metrics_xgb_chosen"]
    c = st.columns(5)
    c[0].metric("PR-AUC", f"{m['pr_auc']:.3f}", help="Area under precision–recall curve; random = fraud rate")
    c[1].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c[2].metric("Precision", f"{m['precision']:.1%}", help="Share of alerts that are real fraud")
    c[3].metric("Recall", f"{m['recall']:.1%}", help="Share of fraud that is caught")
    c[4].metric("F1", f"{m['f1']:.3f}")
    st.caption(f"Decision threshold {m['threshold']:.3f} — {summary['thresholds']['objective']}.")
    st.dataframe(overview, hide_index=True, width="stretch")


def page_analytics() -> None:
    st.title("Fraud analytics")
    st.caption("All figures computed with DuckDB from `sql/fraud_analysis.sql` (train + test periods).")
    cat = read_csv(SQL / "fraud_rate_by_category.csv")
    if cat is None:
        return missing("SQL outputs", "python -m src.run_sql")

    left, right = st.columns(2)
    with left:
        st.subheader("Fraud rate by category")
        st.altair_chart(alt.Chart(cat).mark_bar(color=FRAUD, cornerRadiusEnd=3).encode(
            x=alt.X("fraud_rate_pct:Q", title="Fraud rate (%)"),
            y=alt.Y("category:N", sort="-x", title=None),
            tooltip=["category", "n_transactions", "n_fraud", "fraud_rate_pct", "fraud_amount_usd"],
        ).properties(height=380), width="stretch")
    with right:
        st.subheader("Fraud rate by amount band")
        bands = read_csv(SQL / "amount_band_fraud_rate.csv")
        st.altair_chart(alt.Chart(bands).mark_bar(color=FRAUD, cornerRadiusEnd=3).encode(
            x=alt.X("amount_band:N", title=None, sort=None),
            y=alt.Y("fraud_rate_pct:Q", title="Fraud rate (%)"),
            tooltip=["amount_band", "n_transactions", "n_fraud", "fraud_rate_pct"],
        ).properties(height=380), width="stretch")

    st.subheader("Amount distribution: fraud vs legitimate")
    hist = read_csv(SQL / "amount_histogram_by_status.csv")
    st.altair_chart(alt.Chart(hist).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=alt.X("bin_start_usd:Q", scale=alt.Scale(type="log"), title="Transaction amount ($, log scale)"),
        y=alt.Y("share_within_status:Q", title="Share of transactions within class", axis=alt.Axis(format="%")),
        color=alt.Color("status:N", scale=alt.Scale(domain=["legitimate", "fraud"], range=[LEGIT, FRAUD]), title=None),
        tooltip=["status", "bin_start_usd", "n_transactions", alt.Tooltip("share_within_status:Q", format=".2%")],
    ).properties(height=300), width="stretch")
    # "\$" stops Streamlit's markdown from treating text between two $ signs as LaTeX.
    st.caption("Fraud concentrates at roughly \\$8–25 (small card-testing purchases) and \\$250–1,100; "
               "legitimate spend is mostly under \\$150.")

    left, right = st.columns(2)
    with left:
        st.subheader("Fraud rate by hour of day")
        hour = read_csv(SQL / "fraud_by_hour.csv")
        st.altair_chart(alt.Chart(hour).mark_line(color=FRAUD, point=True).encode(
            x=alt.X("hour_of_day:O", title="Hour"), y=alt.Y("fraud_rate_pct:Q", title="Fraud rate (%)"),
            tooltip=["hour_of_day", "n_transactions", "n_fraud", "fraud_rate_pct"],
        ).properties(height=280), width="stretch")
    with right:
        st.subheader("Fraud rate by weekday")
        dow = read_csv(SQL / "fraud_by_day_of_week.csv")
        st.altair_chart(alt.Chart(dow).mark_bar(color=FRAUD, cornerRadiusEnd=3).encode(
            x=alt.X("day_name:N", sort=DAYS, title=None), y=alt.Y("fraud_rate_pct:Q", title="Fraud rate (%)"),
            tooltip=["day_name", "n_transactions", "n_fraud", "fraud_rate_pct"],
        ).properties(height=280), width="stretch")

    st.subheader("Volume and fraud over time")
    monthly = read_csv(SQL / "monthly_volume_and_fraud.csv")
    monthly["month"] = pd.to_datetime(monthly["month"])
    left, right = st.columns(2)
    left.altair_chart(alt.Chart(monthly).mark_bar(color=LEGIT, cornerRadiusEnd=2).encode(
        x=alt.X("yearmonth(month):T", title=None), y=alt.Y("n_transactions:Q", title="Transactions per month"),
        tooltip=[alt.Tooltip("yearmonth(month):T", title="month"), "n_transactions"],
    ).properties(height=260), width="stretch")
    right.altair_chart(alt.Chart(monthly).mark_line(color=FRAUD, point=True).encode(
        x=alt.X("yearmonth(month):T", title=None), y=alt.Y("fraud_rate_pct:Q", title="Fraud rate (%)"),
        tooltip=[alt.Tooltip("yearmonth(month):T", title="month"), "n_fraud", "fraud_rate_pct"],
    ).properties(height=260), width="stretch")

    st.subheader("Geography: fraud rate by state")
    state = read_csv(SQL / "fraud_rate_by_state.csv")
    st.caption("Read with care: several states contain only 1–4 cardholders, so their rates reflect individual "
               "customers rather than a geographic effect. State was therefore excluded from the model.")
    st.dataframe(state, hide_index=True, width="stretch", height=300)


def page_predictor() -> None:
    st.title("Transaction risk predictor")
    pipe, explainer = load_model()
    if pipe is None:
        return missing("Trained model", "python -m src.train")
    thr = load_thresholds()
    examples = read_csv(config.DASHBOARD_DATA_DIR / "example_transactions.csv")

    st.caption(f"Risk tiers — Low: p < {thr.medium:.3f} · Medium: {thr.medium:.3f} ≤ p < {thr.high:.3f} · "
               f"High: p ≥ {thr.high:.3f}. Medium starts at the alert threshold (precision ≥ 50% on validation), "
               "High where validation precision reaches 90%.")

    preset = None
    if examples is not None:
        options = ["Manual entry"] + [
            f"#{i} · {r.outcome} · ${r.amt:,.2f} {r.category}" for i, r in examples.iterrows()]
        choice = st.selectbox("Start from a real test-period transaction (optional)", options)
        if choice != "Manual entry":
            preset = examples.loc[int(choice.split(" ")[0][1:])]

    def d(key, default):
        if preset is None or pd.isna(preset.get(key, None)):
            return default
        return preset[key]

    categories = sorted(read_csv(SQL / "fraud_rate_by_category.csv")["category"])
    with st.form("txn"):
        st.markdown("**Transaction**")
        c = st.columns(4)
        amt = c[0].number_input("Amount ($)", 1.0, 30000.0, float(d("amt", 850.0)), step=10.0)
        category = c[1].selectbox("Category", categories, index=categories.index(d("category", "shopping_net")))
        hour = c[2].slider("Hour of day", 0, 23, int(d("hour", 23)))
        dow = c[3].selectbox("Day of week", range(7), index=int(d("day_of_week", 4)), format_func=lambda i: DAYS[i])
        st.markdown("**Cardholder**")
        c = st.columns(4)
        age = c[0].number_input("Age", 14.0, 100.0, float(d("age", 52.0)), step=1.0)
        city_pop = c[1].number_input("City population", 20, 3_000_000, int(d("city_pop", 2500)), step=100)
        st.markdown("**Card history up to this transaction**")
        c = st.columns(4)
        prior = c[0].number_input("Previous transactions on card", 0, 10000, int(d("card_prior_txn_count", 600)))
        n1h = c[1].number_input("Transactions in last 1h", 0, 100, int(d("card_txn_count_1h", 0)))
        n24h = c[2].number_input("Transactions in last 24h", 0, 500, int(d("card_txn_count_24h", 6)))
        mins = c[3].number_input("Minutes since previous transaction", 0.0, 50000.0,
                                 float(d("secs_since_prev_txn", 2400.0)) / 60, step=5.0)
        c = st.columns(4)
        mean_amt = c[0].number_input("Card's average amount so far ($)", 0.0, 10000.0, float(d("prior_mean_amt", 65.0)))
        std_amt = c[1].number_input("Std. dev. of card's amounts ($)", 0.0, 10000.0, float(d("prior_std_amt", 80.0)))
        same_cat = c[2].number_input("Previous transactions in this category", 0, 10000,
                                     int(d("prior_same_category_count", 40)))
        submitted = st.form_submit_button("Score transaction", type="primary")

    if not submitted:
        st.info("Fill in the form and press **Score transaction**.")
        return
    if same_cat > prior or n1h > n24h or n24h > prior:
        st.error("Inconsistent history: counts must satisfy 1h ≤ 24h ≤ previous transactions, "
                 "and same-category ≤ previous transactions.")
        return

    X = features_from_card_summary(
        amt=amt, category=category, hour=hour, day_of_week=dow, age=age, city_pop=city_pop,
        prior_txn_count=prior, txn_count_1h=n1h, txn_count_24h=n24h, secs_since_prev_txn=mins * 60,
        prior_mean_amt=mean_amt, prior_std_amt=std_amt, prior_same_category_count=same_cat)
    p = float(pipe.predict_proba(X)[0, 1])
    tier = str(assign_tier(p, thr)[0])
    icon, color = TIER_STYLE[tier]

    c = st.columns([1, 1, 2])
    c[0].metric("Fraud probability", f"{p:.1%}")
    c[1].markdown(f"<div style='font-size:0.9rem;color:{MUTED}'>Risk tier</div>"
                  f"<div style='font-size:2rem;font-weight:600;color:{color}'>{icon} {tier}</div>",
                  unsafe_allow_html=True)
    c[2].markdown({"High": "**Action:** decline / hold automatically (validation precision ≥ 90% in this band).",
                   "Medium": "**Action:** step-up authentication (e.g. one-time passcode) or analyst review.",
                   "Low": "**Action:** approve."}[tier])
    if preset is not None:
        st.caption(f"Actual label of this test transaction: **{'fraud' if preset[config.TARGET] else 'legitimate'}**.")

    st.subheader("Why did the model give this score?")
    sv = explain_rows(pipe, explainer, X)[0]
    fig = plot_waterfall(sv, "Feature contributions (log-odds) for this transaction")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    table = contribution_table(sv)
    table["value"] = table["value"].astype(str)
    st.dataframe(table.head(8), hide_index=True, width="stretch")
    st.caption("SHAP values show how much each input moved the model's score relative to its average prediction. "
               "They explain the model's behaviour — they are **not** causal statements about why fraud occurs.")


def page_performance() -> None:
    st.title("Model performance")
    test = read_csv(config.REPORTS_DIR / "test_metrics.csv")
    comp = read_csv(config.REPORTS_DIR / "model_comparison.csv")
    if test is None or comp is None:
        return missing("Training reports", "python -m src.train")

    st.subheader("Held-out test period (Jun–Dec 2020)")
    cols = ["model", "threshold_rule", "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
    st.dataframe(test[cols].round(4), hide_index=True, width="stretch")
    left, right = st.columns([1, 2])
    with left:
        image("confusion_matrix_xgb_test.png")
    with right:
        image("model_test_pr_roc.png")

    st.subheader("Cross-validation comparison (train, grouped by card)")
    st.dataframe(comp.round(4), hide_index=True, width="stretch")

    st.subheader("Threshold selection (out-of-fold predictions on train)")
    image("threshold_tradeoff.png")
    ttab = read_csv(config.REPORTS_DIR / "threshold_analysis.csv")
    if ttab is not None:
        st.dataframe(ttab.round(4), hide_index=True, width="stretch")

    st.subheader("Risk tiers on the test period")
    tiers = read_csv(config.REPORTS_DIR / "risk_tier_summary.csv")
    if tiers is not None:
        st.dataframe(tiers.round(3), hide_index=True, width="stretch")

    st.subheader("Robustness checks")
    rob = read_csv(config.REPORTS_DIR / "robustness_checks.csv")
    if rob is not None:
        st.dataframe(rob.round(4), hide_index=True, width="stretch")
        st.caption("Checks whether performance depends on the dataset artefact where brand-new cards are always fraud.")

    st.subheader("Global explainability")
    left, right = st.columns(2)
    with left:
        path = config.SHAP_FIGURES_DIR / "shap_importance.png"
        if path.exists():
            st.image(str(path), width="stretch")
    with right:
        path = config.SHAP_FIGURES_DIR / "shap_summary.png"
        if path.exists():
            st.image(str(path), width="stretch")


PAGES = {"Overview": page_overview, "Fraud analytics": page_analytics,
         "Risk predictor": page_predictor, "Model performance": page_performance}

with st.sidebar:
    st.header("🛡️ Fraud Risk Analytics")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Educational portfolio project on synthetic data. Not for real credit decisions.")

PAGES[page]()
