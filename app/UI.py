"""
ui.py — Streamlit UI  (v4)

Pages:
  🔍 Predict        single prediction + optional SHAP explanation waterfall
  📦 Batch          upload CSV, submit batch job, poll, download results
  🔄 Retrain        retrain a model (analyst/admin)
  📋 Versions       view version history, promote any version (admin)
  📊 Models         model catalogue + role access
  🔑 Manage Keys    own keys only
  🛡️ Admin Panel    user management (admin only)
"""
from __future__ import annotations
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

st.markdown("""
<style>
[data-testid="stStatusWidget"] { display: none !important; }
.stButton > button[kind="primary"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
DEFAULTS = {
    "username": "", "password": "", "role": "",
    "api_key": "", "key_prefix": "", "key_expires_at": "",
    "logged_in": False, "has_key": False,
    "last_job_id": "", "last_batch_id": "",
    "pred_result": None, "pred_inputs": {},
    "admin_confirmed": False, "admin_password": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── HTTP helpers ──────────────────────────────────────────────
def _api_online() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except Exception:
        return False

def _req(method, path, **kwargs):
    try:
        return requests.request(method, f"{API_BASE}{path}", timeout=30, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Run: `uvicorn app.main:app --reload`")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out.")
        st.stop()

def _json(r):
    try:
        return r.json()
    except Exception:
        return {"detail": f"HTTP {r.status_code} — no JSON body."}

def _detail(r) -> str:
    d = _json(r).get("detail", f"HTTP {r.status_code}")
    if isinstance(d, dict):
        return d.get("message", str(d))
    if isinstance(d, list):
        return "; ".join(str(x) for x in d)
    return str(d)

def _ah():
    return {"x-api-key": st.session_state.api_key}

ROLE_LABEL = {"admin": "🔴 Admin", "analyst": "🟡 Analyst", "guest": "🟢 Guest"}
def _badge(role): return ROLE_LABEL.get(role, role)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📉 Churn API")
    st.caption("🟢 API online" if _api_online() else "🔴 API offline — start uvicorn")
    st.divider()

    if st.session_state.logged_in and st.session_state.has_key:
        st.write(f"**{st.session_state.username}**")
        st.caption(_badge(st.session_state.role))
        st.caption(f"Key: `{st.session_state.key_prefix}…`")
        if st.session_state.key_expires_at:
            st.caption(f"Expires: {st.session_state.key_expires_at[:10]}")
        st.divider()

        pages = ["🔍 Predict", "📦 Batch", "📊 Models", "🔑 Manage Keys"]
        if st.session_state.role in {"admin", "analyst"}:
            pages.insert(2, "🔄 Retrain")
            pages.insert(3, "📋 Versions")
        if st.session_state.role == "admin":
            pages.append("🛡️ Admin Panel")

        page = st.radio("Navigate", pages)
        st.divider()
        if st.button("🚪 Log out", width=True):
            for k in DEFAULTS:
                st.session_state[k] = DEFAULTS[k]
            st.rerun()
    else:
        page = None

# ═════════════════════════════════════════════════════════════
# GATE 1 — Not logged in
# ═════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.title("Churn Prediction API")
    tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

    with tab_login:
        with st.form("lf"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            ok = st.form_submit_button("Login", type="primary", use_container_width=True)
        if ok:
            r = _req("POST", "/v1/auth/login", json={"username": u, "password": p})
            d = _json(r)
            if r.status_code == 200:
                st.session_state.update(username=d["username"], role=d["role"],
                                        password=p, logged_in=True)
                st.rerun()
            else:
                st.error(_detail(r))

    with tab_signup:
        st.info("Password: 8+ chars, 1 uppercase, 1 digit. Example: `Hello@123`")
        with st.form("sf"):
            su = st.text_input("Username")
            sp = st.text_input("Password", type="password")
            ok2 = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        if ok2:
            r = _req("POST", "/v1/auth/signup", json={"username": su, "password": sp})
            d = _json(r)
            if r.status_code == 201:
                st.session_state.update(username=su, role="guest", password=sp, logged_in=True)
                st.rerun()
            elif r.status_code == 422:
                for e in d.get("details", []):
                    st.error(f"**{e.get('field')}**: {e.get('message')}")
            else:
                st.error(_detail(r))
    st.stop()

# ═════════════════════════════════════════════════════════════
# GATE 2 — No key yet
# ═════════════════════════════════════════════════════════════
if not st.session_state.has_key:
    st.title(f"👋 Hi {st.session_state.username}!")
    st.subheader("Generate your API key to continue")
    ttl = {"admin": "365 days", "analyst": "90 days", "guest": "30 days"}.get(
        st.session_state.role, "30 days")
    st.info(f"Role: **{_badge(st.session_state.role)}** · Key valid for **{ttl}** · Shown once — copy it.")

    col_gen, col_existing = st.columns(2)
    with col_gen:
        with st.form("kgf"):
            kp = st.text_input("Confirm password", type="password", value=st.session_state.password)
            gen = st.form_submit_button("🔑 Generate Key", type="primary", use_container_width=True)
        if gen:
            r = _req("POST", "/v1/auth/keys", json={"username": st.session_state.username, "password": kp})
            d = _json(r)
            if r.status_code == 200:
                st.session_state.update(api_key=d["raw_key"], key_prefix=d["prefix"],
                                        key_expires_at=d["expires_at"], has_key=True, password="")
                st.success("✅ Copy your key now:"); st.code(d["raw_key"], language=None)
                st.rerun()
            else:
                st.error(_detail(r))

    with col_existing:
        st.markdown("**Already have a key?**")
        ex = st.text_input("Paste full key", type="password")
        if st.button("Use this key", use_container_width=True):
            r = _req("GET", "/v1/models", headers={"x-api-key": ex})
            if r.status_code == 200:
                st.session_state.update(api_key=ex, key_prefix=ex[:7], has_key=True, password="")
                st.rerun()
            else:
                st.error("Key rejected.")
    st.stop()

# ═════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════

# ── 🔍 Predict ────────────────────────────────────────────────
if page == "🔍 Predict":
    st.title("🔍 Predict Customer Churn")

    mr = _req("GET", "/v1/models", headers=_ah())
    if mr.status_code == 401:
        st.error("API key expired. Go to **Manage Keys**.")
        st.stop()
    mdata   = _json(mr)
    allowed = mdata.get("your_allowed_models", [])
    amap    = {m["model_id"]: m["description"] for m in mdata.get("available_models", [])}

    if not allowed:
        st.warning("Your role has no model access. Ask an admin to upgrade.")
        st.stop()

    with st.form("predict_form"):
        st.subheader("Customer Features")
        c1, c2, c3 = st.columns(3)
        with c1:
            age    = st.slider("Age", 18, 100, 35)
            tenure = st.slider("Tenure (months)", 0, 120, 24)
            usage  = st.slider("Usage Frequency", 0, 30, 10)
            calls  = st.slider("Support Calls", 0, 20, 2)
        with c2:
            delay   = st.slider("Payment Delay (days)", 0, 30, 5)
            last_int = st.slider("Last Interaction (days ago)", 0, 365, 30)
            spend   = st.number_input("Total Spend ($)", 0.0, 10000.0, 500.0, step=50.0)
        with c3:
            gender   = st.selectbox("Gender", ["Male", "Female"])
            sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            chosen   = st.selectbox("Model", allowed,
                                    format_func=lambda x: f"{x} — {amap.get(x, x)}")
            explain  = st.toggle("Explain prediction (SHAP)", value=False,
                                 help="Shows which features push the prediction up or down. Adds ~5s.")

        submitted = st.form_submit_button("🚀 Run Prediction", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "model_id": chosen, "Age": age, "Gender": gender, "Tenure": tenure,
            "Usage Frequency": usage, "Support Calls": calls, "Payment Delay": delay,
            "Subscription Type": sub_type, "Contract Length": contract,
            "Total Spend": spend, "Last Interaction": last_int,
        }
        with st.spinner("Running prediction…" + (" Computing SHAP values (this takes ~5s)…" if explain else "")):
            pr = _req("POST", f"/v1/predict{'?explain=true' if explain else ''}", json=payload, headers=_ah())
        result = _json(pr)
        if pr.status_code == 200:
            st.session_state.pred_result = result
            st.session_state.pred_inputs = {
                "Age": age, "Gender": gender, "Tenure": tenure,
                "Subscription Type": sub_type, "Model": chosen,
            }
        else:
            st.error(_detail(pr))
            st.session_state.pred_result = None

    if st.session_state.pred_result:
        r    = st.session_state.pred_result
        prob = r["churn_probability"]
        pred = r["churn_prediction"]

        st.divider()
        col_res, col_detail = st.columns([1, 2])

        with col_res:
            if pred == 1:
                st.error("⚠️ **Likely to Churn**")
            else:
                st.success("✅ **Likely to Stay**")
            st.metric("Churn Probability", f"{prob:.1%}")
            st.progress(prob)
            v = r.get("model_version")
            st.caption(f"Model: `{r['model_id']}` · Version: `{v if v else 'unknown'}`")

        with col_detail:
            exp = r.get("explanation")
            if exp:
                st.subheader("Feature contributions (SHAP)")
                st.caption(
                    "🔴 Red = pushes toward churn  ·  🟢 Green = pushes away. "
                    "Longer bar = stronger influence."
                )
                import pandas as pd
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                df_exp  = pd.DataFrame(exp)
                df_exp  = df_exp[df_exp["feature"] != "explanation_error"]
                df_exp  = df_exp.sort_values("shap_value")
                n       = len(df_exp)
                colours = ["#e74c3c" if v > 0 else "#27ae60" for v in df_exp["shap_value"]]

                fig, ax = plt.subplots(figsize=(6, max(3, n * 0.45)))
                bars = ax.barh(df_exp["feature"], df_exp["shap_value"],
                               color=colours, edgecolor="none", height=0.6)

                for bar, val in zip(bars, df_exp["shap_value"]):
                    pad = 0.001
                    ha  = "left" if val >= 0 else "right"
                    x   = val + pad if val >= 0 else val - pad
                    ax.text(x, bar.get_y() + bar.get_height() / 2,
                            f"{val:+.4f}", va="center", ha=ha, fontsize=8.5)

                ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
                ax.set_xlabel("SHAP value  (impact on churn probability)", fontsize=9)
                ax.tick_params(axis="y", labelsize=9)
                ax.tick_params(axis="x", labelsize=8)
                ax.set_facecolor("#f9f9f9")
                fig.patch.set_facecolor("#f9f9f9")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                with st.expander("View as table"):
                    st.dataframe(
                        df_exp[["feature","shap_value","direction"]]
                        .rename(columns={"feature":"Feature",
                                         "shap_value":"SHAP value",
                                         "direction":"Effect"})
                        .reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                    )

                err_rows = [x for x in exp if x.get("feature") == "explanation_error"]
                if err_rows:
                    st.warning(f"SHAP note: {err_rows[0].get('direction')}")

            else:
                st.markdown("**Inputs used**")
                for k, v in st.session_state.pred_inputs.items():
                    st.write(f"- **{k}:** {v}")


# ── 📦 Batch ──────────────────────────────────────────────────
elif page == "📦 Batch":
    st.title("📦 Batch Prediction")
    st.info(
        "Upload a CSV with customer feature columns (no `Churn` column needed). "
        "The results CSV will have all original columns plus `churn_prediction` and `churn_probability`."
    )

    mr      = _req("GET", "/v1/models", headers=_ah())
    allowed = _json(mr).get("your_allowed_models", []) if mr.status_code == 200 else []
    amap    = {m["model_id"]: m["description"] for m in _json(mr).get("available_models", [])}

    col_up, col_status = st.columns(2)

    with col_up:
        st.subheader("Submit a job")
        model_id = st.selectbox("Model", allowed, format_func=lambda x: f"{x} — {amap.get(x, x)}")
        uploaded = st.file_uploader("Upload CSV (max 1,000 rows)", type=["csv"])

        if uploaded:
            import pandas as pd
            df = pd.read_csv(uploaded)
            # Drop Churn column if present — not needed for prediction
            if "Churn" in df.columns:
                df = df.drop(columns=["Churn"])
                st.caption("ℹ️ `Churn` column removed — not needed for prediction.")

            st.dataframe(df.head(3), use_container_width=True)
            st.caption(f"{len(df)} rows · {len(df.columns)} columns")

            if len(df) > 1000:
                st.error("Maximum 1,000 rows per batch. Please split the file.")
            else:
                if st.button("🚀 Submit Batch Job", type="primary", use_container_width=True):
                    records = df.to_dict("records")
                    r = _req("POST", "/v1/predict/batch",
                             json={"model_id": model_id, "data": records}, headers=_ah())
                    d = _json(r)
                    if r.status_code == 202:
                        st.success(f"✅ Job queued! ID: `{d['job_id']}`")
                        st.session_state.last_batch_id = d["job_id"]
                    else:
                        st.error(_detail(r))

    with col_status:
        st.subheader("Poll & download")
        jid = st.text_input("Job ID", value=st.session_state.last_batch_id)

        col_a, col_b = st.columns(2)
        refresh = col_a.button("🔄 Refresh", use_container_width=True)
        download_clicked = col_b.button("⬇️ Download CSV", use_container_width=True)

        if refresh and jid:
            r = _req("GET", f"/v1/predict/batch/{jid}", headers=_ah())
            d = _json(r)
            if r.status_code == 200:
                s    = d["status"]
                icon = {"queued": "🟡", "running": "🔵", "done": "🟢", "failed": "🔴"}.get(s, "⚪")
                st.write(f"**Status:** {icon} {s.title()}")
                st.write(f"Model: `{d['model_id']}` v{d.get('model_version','?')} · "
                         f"Rows: {d['processed_rows']}/{d['total_rows']}")
                if d.get("churn_rate") is not None:
                    st.metric("Churn Rate in Batch", f"{d['churn_rate']:.1%}")
                    st.metric("Avg Churn Probability", f"{d['avg_probability']:.1%}")
                if d.get("finished_at"):
                    st.caption(f"Finished: {d['finished_at'][:19]}")
                if d.get("error"):
                    st.error(d["error"])
            else:
                st.error(_detail(r))

        if download_clicked and jid:
            r = _req("GET", f"/v1/predict/batch/{jid}/download", headers=_ah())
            if r.status_code == 200:
                st.download_button(
                    "📥 Save results CSV",
                    data=r.content,
                    file_name=f"batch_{jid[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.error(_detail(r))


# ── 🔄 Retrain ────────────────────────────────────────────────
elif page == "🔄 Retrain":
    st.title("🔄 Retrain a Model")

    if st.session_state.role not in {"admin", "analyst"}:
        st.error("Requires analyst or admin role.")
        st.stop()

    st.info(
        "Upload a CSV with feature columns + a `Churn` column (0 or 1). "
        "Each successful retrain creates a **new versioned artifact** and promotes it to active. "
        "You can roll back at any time in **📋 Versions**."
    )

    mr      = _req("GET", "/v1/models", headers=_ah())
    allowed = _json(mr).get("your_allowed_models", []) if mr.status_code == 200 else []
    model_id = st.selectbox("Model to retrain", allowed or ["rf"])
    notes    = st.text_input("Version notes (optional)", placeholder="e.g. Added Q2 2025 data")
    uploaded = st.file_uploader("Training CSV", type=["csv"])

    if uploaded:
        import pandas as pd
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(3), use_container_width=True)
        st.caption(f"{len(df)} rows · {len(df.columns)} columns")

        if "Churn" not in df.columns:
            st.error("CSV must have a `Churn` column (0 or 1).")
        else:
            if st.button("🚀 Submit Retrain Job", type="primary"):
                payload = {
                    "model_id": model_id,
                    "notes":    notes or None,
                    "data":     df[[c for c in df.columns if c != "Churn"]].to_dict("records"),
                    "labels":   df["Churn"].tolist(),
                }
                with st.spinner("Submitting…"):
                    r = _req("POST", "/v1/retrain", json=payload, headers=_ah())
                d = _json(r)
                if r.status_code == 202:
                    st.success(f"Job queued! ID: `{d['job_id']}`")
                    st.session_state.last_job_id = d["job_id"]
                else:
                    st.error(_detail(r))

    st.divider()
    st.subheader("Poll job status")
    jid = st.text_input("Job ID", value=st.session_state.last_job_id)
    if st.button("🔄 Refresh") and jid:
        r = _req("GET", f"/v1/retrain/{jid}", headers=_ah())
        d = _json(r)
        if r.status_code == 200:
            s    = d["status"]
            icon = {"queued": "🟡", "running": "🔵", "done": "🟢", "failed": "🔴"}.get(s, "⚪")
            st.write(f"**Status:** {icon} {s.title()}")
            if d.get("new_version"):
                st.success(f"New version created: **v{d['new_version']}** (now active)")
            if d.get("metrics"):
                st.json(d["metrics"])
            if d.get("error"):
                st.error(d["error"])
        else:
            st.error(_detail(r))


# ── 📋 Versions ───────────────────────────────────────────────
elif page == "📋 Versions":
    st.title("📋 Model Version History")

    mr      = _req("GET", "/v1/models", headers=_ah())
    allowed = _json(mr).get("your_allowed_models", []) if mr.status_code == 200 else []
    model_id = st.selectbox("Select model", allowed or ["rf"])

    if st.button("Load versions"):
        r = _req("GET", f"/v1/models/{model_id}/versions", headers=_ah())
        if r.status_code == 200:
            versions = _json(r)
            if not versions:
                st.info("No versions found.")
            else:
                import pandas as pd
                rows = []
                for v in versions:
                    m = v.get("metrics") or {}
                    rows.append({
                        "Version":    v["version_number"],
                        "Active":     "✅" if v["is_active"] else "",
                        "Trained by": v.get("trained_by") or "—",
                        "Rows":       v.get("train_rows") or "—",
                        "AUC":        m.get("roc_auc", "—"),
                        "F1":         m.get("f1", "—"),
                        "Accuracy":   m.get("accuracy", "—"),
                        "Notes":      v.get("notes") or "—",
                        "Created":    v["created_at"][:10],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.error(_detail(r))

    if st.session_state.role == "admin":
        st.divider()
        st.subheader("Promote a version")
        st.caption(
            "Sets the chosen version as active. Immediately hot-swaps the in-memory model "
            "— no restart needed. Use this to roll back if a retrain degraded quality."
        )
        with st.form("promote_form"):
            ver_num = st.number_input("Version number to promote", min_value=1, step=1, value=1)
            go      = st.form_submit_button("Promote", type="primary")
        if go:
            r = _req("POST", f"/v1/models/{model_id}/promote",
                     json={"version_number": int(ver_num)}, headers=_ah())
            d = _json(r)
            if r.status_code == 200:
                st.success(d.get("message", "Promoted."))
            else:
                st.error(_detail(r))


# ── 📊 Models ─────────────────────────────────────────────────
elif page == "📊 Models":
    st.title("📊 Models")
    mr = _req("GET", "/v1/models", headers=_ah())
    if mr.status_code == 200:
        data    = _json(mr)
        allowed = set(data.get("your_allowed_models", []))
        for m in data.get("available_models", []):
            mid = m["model_id"]
            with st.container(border=True):
                a, b, c = st.columns([1, 3, 2])
                a.code(mid)
                b.write(m["description"])
                c.write("✅ Accessible" if mid in allowed else "🔒 No access")
    st.divider()
    rr = _req("GET", "/v1/auth/roles")
    if rr.status_code == 200:
        st.subheader("Role permissions")
        for role, models in _json(rr).items():
            st.write(f"**{role.title()}:** {', '.join(f'`{m}`' for m in models)}")


# ── 🔑 Manage Keys ────────────────────────────────────────────
elif page == "🔑 Manage Keys":
    st.title("🔑 Manage Your API Keys")
    st.write(f"**Active key prefix:** `{st.session_state.key_prefix}`")
    if st.session_state.key_expires_at:
        st.write(f"**Expires:** {st.session_state.key_expires_at[:10]}")

    col_new, col_rev = st.columns(2)
    with col_new:
        st.subheader("Generate new key")
        with st.form("nkf"):
            np_ = st.text_input("Confirm password", type="password")
            go  = st.form_submit_button("Generate", type="primary", use_container_width=True)
        if go:
            r = _req("POST", "/v1/auth/keys",
                     json={"username": st.session_state.username, "password": np_})
            d = _json(r)
            if r.status_code == 200:
                st.success("✅ Copy now:"); st.code(d["raw_key"], language=None)
                st.session_state.update(api_key=d["raw_key"], key_prefix=d["prefix"],
                                        key_expires_at=d["expires_at"])
            else:
                st.error(_detail(r))

    with col_rev:
        st.subheader("Revoke a key")
        with st.form("rvf"):
            rp_  = st.text_input("Confirm password", type="password")
            rpfx = st.text_input("Key prefix to revoke")
            rev  = st.form_submit_button("Revoke", type="primary", use_container_width=True)
        if rev:
            r = _req("DELETE", f"/v1/auth/keys/{rpfx}",
                     json={"username": st.session_state.username, "password": rp_})
            d = _json(r)
            if r.status_code == 200:
                st.success(d.get("message", "Revoked."))
                if rpfx.startswith(st.session_state.key_prefix):
                    for k in DEFAULTS:
                        st.session_state[k] = DEFAULTS[k]
                    st.rerun()
            else:
                st.error(_detail(r))


# ── 🛡️ Admin Panel ────────────────────────────────────────────
elif page == "🛡️ Admin Panel":
    st.title("🛡️ Admin Panel")

    if not st.session_state.admin_confirmed:
        st.warning("Confirm admin password to proceed.")
        with st.form("acp"):
            ap = st.text_input("Admin password", type="password")
            ok = st.form_submit_button("Confirm", type="primary")
        if ok:
            r = _req("POST", "/v1/auth/login",
                     json={"username": st.session_state.username, "password": ap})
            if r.status_code == 200 and _json(r).get("role") == "admin":
                st.session_state.admin_confirmed = True
                st.session_state.admin_password  = ap
                st.rerun()
            else:
                st.error("Wrong password or insufficient role.")
        st.stop()

    tab_users, tab_roles, tab_newuser = st.tabs(["👥 Users", "🔐 Change Role", "➕ Create User"])

    with tab_users:
        st.subheader("All Users")
        if st.button("🔄 Refresh", key="ru"):
            st.session_state.pop("_users", None)
        if "_users" not in st.session_state:
            r = _req("GET", "/v1/auth/users", params={
                "admin_username": st.session_state.username,
                "admin_password": st.session_state.admin_password,
            })
            st.session_state["_users"] = _json(r) if r.status_code == 200 else []

        users = st.session_state["_users"]
        if users:
            import pandas as pd
            st.dataframe(pd.DataFrame(users)[["id","username","role","is_active","created_at"]],
                         use_container_width=True, hide_index=True)

    with tab_roles:
        users = st.session_state.get("_users", [])
        unames = [u["username"] for u in users if u["username"] != st.session_state.username]
        if not unames:
            st.info("Load Users tab first.")
        else:
            with st.form("rrf"):
                target   = st.selectbox("User", unames)
                new_role = st.selectbox("New role", ["guest", "analyst", "admin"])
                go       = st.form_submit_button("Update Role", type="primary", use_container_width=True)
            if go:
                r = _req("PATCH", f"/v1/auth/users/{target}/role",
                         params={"admin_username": st.session_state.username,
                                 "admin_password": st.session_state.admin_password},
                         json={"role": new_role})
                if r.status_code == 200:
                    st.success(f"✅ {target} → `{new_role}`")
                    st.session_state.pop("_users", None)
                else:
                    st.error(_detail(r))

    with tab_newuser:
        st.info("Create analyst or admin accounts directly — bypasses the public sign-up form.")
        with st.form("cuf"):
            nu  = st.text_input("Username")
            np2 = st.text_input("Temporary password", type="password")
            nr  = st.selectbox("Role", ["guest", "analyst", "admin"])
            go3 = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        if go3:
            r = _req("POST", "/v1/auth/signup", json={"username": nu, "password": np2})
            d = _json(r)
            if r.status_code == 201:
                if nr != "guest":
                    r2 = _req("PATCH", f"/v1/auth/users/{nu}/role",
                              params={"admin_username": st.session_state.username,
                                      "admin_password": st.session_state.admin_password},
                              json={"role": nr})
                    if r2.status_code != 200:
                        st.warning(f"Account created but promotion failed: {_detail(r2)}")
                        st.stop()
                st.success(f"✅ Account **{nu}** created with role `{nr}`.")
                st.caption(f"Username: `{nu}` · Password: `{np2}` *(temporary)*")
                st.session_state.pop("_users", None)
            elif r.status_code == 409:
                st.error("Username already taken.")
            elif r.status_code == 422:
                for e in d.get("details", []):
                    st.error(f"**{e.get('field')}**: {e.get('message')}")
            else:
                st.error(_detail(r))