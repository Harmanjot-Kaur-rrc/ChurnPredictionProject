"""
app/ui.py
─────────
Interactive Streamlit UI for the Churn Prediction API.
Run with: streamlit run app/ui.py
"""
import streamlit as st
import requests
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight · Prediction Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --border:    #252a35;
    --accent:    #00e5ff;
    --accent2:   #ff4d6d;
    --accent3:   #a8ff78;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Typography ── */
h1, h2, h3 { font-family: var(--mono) !important; letter-spacing: -0.02em; }

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.12) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* ── Form submit button ── */
[data-testid="stFormSubmitButton"] button {
    background: var(--accent) !important;
    border: none !important;
    color: var(--bg) !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    width: 100% !important;
    padding: 0.75rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--mono) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--surface) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent3), var(--accent)) !important;
}

/* ── Selectbox dropdown ── */
[data-baseweb="popover"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Result card ── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-card.churn {
    border-left: 3px solid var(--accent2);
}
.result-card.stay {
    border-left: 3px solid var(--accent3);
}

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.25);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 0.05em;
}
.model-badge.locked {
    background: rgba(100,116,139,0.08);
    border-color: rgba(100,116,139,0.25);
    color: var(--muted);
}

/* ── Tag label ── */
.tag {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Risk bar label ── */
.risk-label {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for key, default in {
    "api_key": "",
    "user_role": "",
    "allowed_models": [],
    "all_models": [],
    "authenticated": False,
    "last_result": None,
    "history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def call_models(api_key: str) -> dict | None:
    try:
        r = requests.get(
            f"{API_BASE}/v1/models",
            headers={"x-api-key": api_key},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
        return {"error": r.status_code, "detail": r.json().get("detail", "Unknown")}
    except requests.exceptions.ConnectionError:
        return {"error": "connection", "detail": "Cannot reach API on localhost:8000"}


def call_predict(payload: dict, api_key: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/v1/predict",
            json=payload,
            headers={"x-api-key": api_key},
            timeout=10,
        )
        return {"status": r.status_code, "body": r.json()}
    except requests.exceptions.ConnectionError:
        return {"status": 0, "body": {"detail": "Cannot reach API on localhost:8000"}}


def risk_color(prob: float) -> str:
    if prob >= 0.7:
        return "#ff4d6d"
    if prob >= 0.4:
        return "#fbbf24"
    return "#a8ff78"


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH RISK"
    if prob >= 0.4:
        return "MEDIUM RISK"
    return "LOW RISK"


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Auth + Status
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 18px; font-weight: 700; color: #00e5ff; letter-spacing: -0.02em;'>◈ ChurnSight</div>
        <div style='font-family: Space Mono, monospace; font-size: 10px; color: #64748b; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 2px;'>Prediction Console v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tag">Authentication</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)

    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter your x-api-key…",
        label_visibility="collapsed",
        value=st.session_state.api_key,
    )

    connect_btn = st.button("→ Connect", use_container_width=True)

    if connect_btn and api_key_input:
        with st.spinner("Authenticating…"):
            result = call_models(api_key_input)

        if result and "error" not in result:
            st.session_state.api_key = api_key_input
            st.session_state.authenticated = True
            st.session_state.all_models = result["available_models"]
            st.session_state.allowed_models = result["your_allowed_models"]
            # Infer role from allowed model count
            n = len(result["your_allowed_models"])
            st.session_state.user_role = "admin" if n == 5 else "analyst" if n == 3 else "guest"
        elif result and result.get("error") == "connection":
            st.error("⚡ API offline — start uvicorn first")
        else:
            st.error(f"✕ Auth failed ({result.get('error', '?')})")
            st.session_state.authenticated = False

    if connect_btn and not api_key_input:
        st.warning("Enter an API key first.")

    # ── Status panel ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="tag">Session Status</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    if st.session_state.authenticated:
        st.markdown(f"""
        <div style='font-family: Space Mono, monospace; font-size: 12px; line-height: 2;'>
            <span style='color: #a8ff78;'>●</span> <span style='color: #64748b;'>STATUS</span> &nbsp; Connected<br>
            <span style='color: #00e5ff;'>◆</span> <span style='color: #64748b;'>ROLE</span> &nbsp;&nbsp;&nbsp; {st.session_state.user_role.upper()}<br>
            <span style='color: #fbbf24;'>◇</span> <span style='color: #64748b;'>MODELS</span> &nbsp; {len(st.session_state.allowed_models)} available
        </div>
        """, unsafe_allow_html=True)

        if st.button("✕ Disconnect", use_container_width=True):
            for k in ["api_key", "user_role", "allowed_models", "all_models", "last_result"]:
                st.session_state[k] = [] if k in ["allowed_models", "all_models", "history"] else ""
            st.session_state.authenticated = False
            st.rerun()
    else:
        st.markdown("""
        <div style='font-family: Space Mono, monospace; font-size: 12px;'>
            <span style='color: #64748b;'>● DISCONNECTED</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Demo keys hint ──
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("Demo API Keys"):
        st.markdown("""
        <div style='font-family: Space Mono, monospace; font-size: 11px; line-height: 2; color: #94a3b8;'>
            <b style='color:#00e5ff;'>admin-key-123</b><br>
            &nbsp;→ all 5 models<br><br>
            <b style='color:#00e5ff;'>analyst-key-456</b><br>
            &nbsp;→ logreg, rf, xgb<br><br>
            <b style='color:#00e5ff;'>guest-key-789</b><br>
            &nbsp;→ logreg only
        </div>
        """, unsafe_allow_html=True)

    # ── Prediction history ──
    if st.session_state.history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="tag">Recent Predictions</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            color = "#ff4d6d" if h["churn"] else "#a8ff78"
            label = "CHURN" if h["churn"] else "STAY"
            st.markdown(f"""
            <div style='font-family: Space Mono, monospace; font-size: 11px;
                        padding: 6px 8px; margin-bottom: 4px;
                        border-left: 2px solid {color};
                        background: rgba(255,255,255,0.02);'>
                <span style='color:{color};'>{label}</span>
                <span style='color:#64748b;'> · {h["prob"]}% · {h["model"]}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    # ── Landing state ──
    st.markdown("""
    <div style='text-align: center; padding: 6rem 2rem 4rem 2rem;'>
        <div style='font-family: Space Mono, monospace; font-size: 48px; font-weight: 700;
                    color: #00e5ff; letter-spacing: -0.03em; line-height: 1;'>◈</div>
        <h1 style='font-family: Space Mono, monospace; font-size: 32px; font-weight: 700;
                   color: #e2e8f0; letter-spacing: -0.03em; margin: 1rem 0 0.5rem 0;'>
            ChurnSight
        </h1>
        <p style='font-family: DM Sans, sans-serif; color: #64748b; font-size: 16px;
                  max-width: 400px; margin: 0 auto 2rem auto; font-weight: 300;'>
            ML-powered customer churn prediction.<br>
            Authenticate with your API key to begin.
        </p>
        <div style='font-family: Space Mono, monospace; font-size: 12px; color: #334155;
                    letter-spacing: 0.1em;'>
            ← Enter your key in the sidebar
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Header ──
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h1 style='font-family: Space Mono, monospace; font-size: 22px;
                   font-weight: 700; color: #e2e8f0; margin: 0; letter-spacing: -0.02em;'>
            Churn Prediction Console
        </h1>
        <div style='font-family: Space Mono, monospace; font-size: 11px;
                    color: #64748b; letter-spacing: 0.1em; margin-top: 4px;'>
            STEP 01 — SELECT MODEL &nbsp;·&nbsp; STEP 02 — ENTER DATA &nbsp;·&nbsp; STEP 03 — PREDICT
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model selection ──
    st.markdown('<div class="tag">Step 01 · Model Selection</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # Show all models as badges
    badge_html = ""
    for m in st.session_state.all_models:
        authorized = m["model_id"] in st.session_state.allowed_models
        css_class = "model-badge" if authorized else "model-badge locked"
        icon = "" if authorized else "🔒 "
        badge_html += f'<span class="{css_class}">{icon}{m["model_id"]} · {m["description"]}</span> &nbsp;'

    st.markdown(f"<div style='margin-bottom: 12px;'>{badge_html}</div>", unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Choose model",
        options=st.session_state.allowed_models,
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Customer data form ──
    st.markdown('<div class="tag">Step 02 · Customer Data</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    with st.form("predict_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="tag">Demographics</div>', unsafe_allow_html=True)
            age = st.number_input("Age", min_value=18, max_value=100, value=35,
                                  help="Customer age (18–100)")
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=24,
                                     help="Months as a customer (0–120)")

        with col2:
            st.markdown('<div class="tag">Behaviour</div>', unsafe_allow_html=True)
            usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=30, value=10,
                                               help="Service uses per month (0–30)")
            support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2,
                                            help="Support calls last month (0–20)")
            payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=30, value=5,
                                            help="Days payment was delayed (0–30)")
            last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=30,
                                               help="Days since last activity (0–365)")

        with col3:
            st.markdown('<div class="tag">Account</div>', unsafe_allow_html=True)
            subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0,
                                          value=500.0, step=50.0,
                                          help="Total USD spent (0–10,000)")

        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(f"◈  Run Prediction  ·  {selected_model.upper()}")

    # ─────────────────────────────────────────────────────────
    # PREDICTION RESULT
    # ─────────────────────────────────────────────────────────
    if submitted:
        payload = {
            "model_id": selected_model,
            "Age": age,
            "Gender": gender,
            "Tenure": tenure,
            "Usage Frequency": usage_frequency,
            "Support Calls": support_calls,
            "Payment Delay": payment_delay,
            "Subscription Type": subscription_type,
            "Contract Length": contract_length,
            "Total Spend": total_spend,
            "Last Interaction": last_interaction,
        }

        with st.spinner("Running inference…"):
            time.sleep(0.3)  # brief pause so spinner is visible
            response = call_predict(payload, st.session_state.api_key)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="tag">Step 03 · Result</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        if response["status"] == 200:
            result = response["body"]
            churn = result["churn_prediction"]
            prob = result["churn_probability"]
            pct = round(prob * 100, 1)
            color = risk_color(prob)
            label = risk_label(prob)
            card_class = "churn" if churn else "stay"
            verdict = "WILL CHURN" if churn else "WILL STAY"
            verdict_color = "#ff4d6d" if churn else "#a8ff78"

            # Save to history
            st.session_state.history.append({
                "churn": bool(churn),
                "prob": pct,
                "model": selected_model,
            })

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Verdict", verdict)
            m2.metric("Churn Probability", f"{pct}%")
            m3.metric("Risk Level", label)
            m4.metric("Model", selected_model.upper())

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

            # Risk bar
            st.markdown(f'<div class="risk-label">Risk Score &nbsp; {pct}%</div>', unsafe_allow_html=True)
            st.progress(prob)

            # Interpretation card
            if prob >= 0.7:
                st.error(
                    f"⚠  **High churn risk ({pct}%).** "
                    "Immediate retention action recommended — consider a personalised offer or account review."
                )
            elif prob >= 0.4:
                st.warning(
                    f"◆  **Medium churn risk ({pct}%).** "
                    "Proactive outreach advised within the next 30 days."
                )
            else:
                st.success(
                    f"✓  **Low churn risk ({pct}%).** "
                    "Customer appears stable — standard engagement is sufficient."
                )

            # Input summary
            with st.expander("View submitted data"):
                cols = st.columns(5)
                fields = [
                    ("Age", age), ("Gender", gender), ("Tenure", f"{tenure} mo"),
                    ("Usage Freq.", usage_frequency), ("Support Calls", support_calls),
                    ("Payment Delay", f"{payment_delay}d"), ("Subscription", subscription_type),
                    ("Contract", contract_length), ("Total Spend", f"${total_spend:,.0f}"),
                    ("Last Interaction", f"{last_interaction}d"),
                ]
                for i, (label_text, val) in enumerate(fields):
                    cols[i % 5].metric(label_text, val)

        elif response["status"] == 422:
            st.error("✕ Validation Error")
            details = response["body"].get("details", [])
            for d in details:
                st.markdown(
                    f"<div style='font-family: Space Mono, monospace; font-size: 12px; "
                    f"color: #ff4d6d; padding: 4px 0;'>"
                    f"  <b>{d.get('field', '?')}</b> — {d.get('message', '')}</div>",
                    unsafe_allow_html=True,
                )

        elif response["status"] == 403:
            detail = response["body"].get("detail", {})
            msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
            st.error(f"✕ Not Authorized: {msg}")

        elif response["status"] == 503:
            st.error("✕ Model not loaded. Run `python src/train_pipeline.py` first.")

        elif response["status"] == 0:
            st.error("✕ Cannot reach API. Is `uvicorn app.main:app --reload` running?")

        else:
            st.error(f"✕ Unexpected error ({response['status']}): {response['body']}")