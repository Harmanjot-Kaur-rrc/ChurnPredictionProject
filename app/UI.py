"""
ui.py — Streamlit UI for Churn Prediction API

Auth gates:
  1. Not logged in          → Login / Sign Up only
  2. Logged in, no key      → Generate Key only
  3. Logged in + key        → Full app

Pages (after auth):
  🔍 Predict       — all roles
  🔄 Retrain       — analyst + admin only
  📊 Models        — all roles
  🔑 Manage Keys   — all roles (own keys only)
  🛡️ Admin Panel   — admin only (user management, role assignment)
"""
from __future__ import annotations
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

# ─────────────────────────────────────────────────────────────
# Custom CSS — removes the running-man spinner on widget changes,
# styles the app cleanly
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide the stale-state running indicator on sliders/selects */
[data-testid="stStatusWidget"] { display: none !important; }

/* Subtle card containers */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: transparent;
}

/* Make primary buttons full-width look better */
.stButton > button[kind="primary"] {
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* Sidebar role badge */
.role-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
DEFAULTS: dict = {
    "username":       "",
    "password":       "",
    "role":           "",
    "api_key":        "",
    "key_prefix":     "",
    "key_expires_at": "",
    "logged_in":      False,
    "has_key":        False,
    "last_job_id":    "",
    # Prediction form — stored here so sliders don't re-trigger API calls
    "pred_result":    None,
    "pred_inputs":    {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────
def _api_online() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except Exception:
        return False


def _req(method: str, path: str, **kwargs) -> requests.Response:
    try:
        return requests.request(method, f"{API_BASE}{path}", timeout=15, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ **Cannot reach the API.**  \n"
            "Start it in a separate terminal:  \n"
            "```\nuvicorn app.main:app --reload\n```"
        )
        st.stop()
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out — try again.")
        st.stop()


def _json(r: requests.Response) -> dict:
    try:
        return r.json()
    except Exception:
        return {"detail": f"HTTP {r.status_code} — no JSON body. Check uvicorn logs."}


def _detail(r: requests.Response) -> str:
    d = _json(r).get("detail", f"HTTP {r.status_code}")
    if isinstance(d, dict):
        return d.get("message", str(d))
    if isinstance(d, list):
        return "; ".join(str(x) for x in d)
    return str(d)


def _ah() -> dict:
    """Auth headers for current session key."""
    return {"x-api-key": st.session_state.api_key}


def _admin_ah() -> dict:
    """Auth headers using the stored admin key (for admin-panel calls)."""
    return {"x-api-key": st.session_state.get("admin_key", "")}


ROLE_COLOUR = {"admin": "#e74c3c", "analyst": "#f39c12", "guest": "#27ae60"}
ROLE_LABEL  = {"admin": "🔴 Admin", "analyst": "🟡 Analyst", "guest": "🟢 Guest"}


def _badge(role: str) -> str:
    return ROLE_LABEL.get(role, f"⚪ {role}")


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📉 Churn API")
    online = _api_online()
    st.caption("🟢 API online" if online else "🔴 API offline — start uvicorn")
    st.divider()

    if st.session_state.logged_in and st.session_state.has_key:
        st.markdown(f"**{st.session_state.username}**")
        st.caption(_badge(st.session_state.role))
        st.caption(f"Key: `{st.session_state.key_prefix}…`")
        if st.session_state.key_expires_at:
            st.caption(f"Expires: {st.session_state.key_expires_at[:10]}")
        st.divider()

        # Build page list based on role
        pages = ["🔍 Predict", "📊 Models", "🔑 Manage Keys"]
        if st.session_state.role in {"admin", "analyst"}:
            pages.insert(1, "🔄 Retrain")
        if st.session_state.role == "admin":
            pages.append("🛡️ Admin Panel")

        page = st.radio("Navigate", pages)

        st.divider()
        if st.button("🚪 Log out", use_container_width=True):
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
    st.caption("Sign in or create an account to get started.")

    tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

    with tab_login:
        with st.form("login_form"):
            u  = st.text_input("Username")
            p  = st.text_input("Password", type="password")
            ok = st.form_submit_button("Login", type="primary", use_container_width=True)
        if ok:
            if not u or not p:
                st.warning("Enter both fields.")
            else:
                r    = _req("POST", "/v1/auth/login", json={"username": u, "password": p})
                data = _json(r)
                if r.status_code == 200:
                    st.session_state.username  = data["username"]
                    st.session_state.role      = data["role"]
                    st.session_state.password  = p
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error(_detail(r))

    with tab_signup:
        st.info("Password: 8+ chars, 1 uppercase, 1 digit. Example: `Hello@123`")
        with st.form("signup_form"):
            su  = st.text_input("Username")
            sp  = st.text_input("Password", type="password")
            ok2 = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        if ok2:
            if not su or not sp:
                st.warning("Fill in both fields.")
            else:
                r    = _req("POST", "/v1/auth/signup", json={"username": su, "password": sp})
                data = _json(r)
                if r.status_code == 201:
                    st.session_state.username  = su
                    st.session_state.role      = "guest"
                    st.session_state.password  = sp
                    st.session_state.logged_in = True
                    st.rerun()
                elif r.status_code == 422:
                    for d in data.get("details", []):
                        st.error(f"**{d.get('field')}**: {d.get('message')}")
                else:
                    st.error(_detail(r))
    st.stop()


# ═════════════════════════════════════════════════════════════
# GATE 2 — Logged in but no key yet
# ═════════════════════════════════════════════════════════════
if not st.session_state.has_key:
    st.title(f"👋 Hi {st.session_state.username}!")
    st.subheader("Generate your API key to continue")

    ttl = {"admin": "365 days", "analyst": "90 days", "guest": "30 days"}.get(
        st.session_state.role, "30 days"
    )
    st.info(
        f"Role: **{_badge(st.session_state.role)}** · Key valid for **{ttl}**  \n"
        "The raw key is shown **once** — copy it before leaving this page."
    )

    col_gen, col_existing = st.columns(2)

    with col_gen:
        st.markdown("**Generate a new key**")
        with st.form("keygen_gate"):
            kp  = st.text_input("Confirm password", type="password",
                                value=st.session_state.password)
            gen = st.form_submit_button("🔑 Generate Key", type="primary",
                                        use_container_width=True)
        if gen:
            r    = _req("POST", "/v1/auth/keys",
                        json={"username": st.session_state.username, "password": kp})
            data = _json(r)
            if r.status_code == 200:
                st.session_state.api_key        = data["raw_key"]
                st.session_state.key_prefix     = data["prefix"]
                st.session_state.key_expires_at = data["expires_at"]
                st.session_state.has_key        = True
                st.session_state.password       = ""
                st.success("✅ Key generated — copy it now:")
                st.code(data["raw_key"], language=None)
                st.caption(f"Expires: {data['expires_at'][:10]}")
                st.rerun()
            else:
                st.error(_detail(r))

    with col_existing:
        st.markdown("**Already have a key?**")
        existing = st.text_input("Paste full key", type="password")
        if st.button("Use this key", use_container_width=True):
            if existing:
                r = _req("GET", "/v1/models", headers={"x-api-key": existing})
                if r.status_code == 200:
                    st.session_state.api_key    = existing
                    st.session_state.key_prefix = existing[:7]
                    st.session_state.has_key    = True
                    st.session_state.password   = ""
                    st.rerun()
                elif r.status_code == 401:
                    err = _json(r).get("detail", {})
                    code = err.get("error", "") if isinstance(err, dict) else ""
                    if code == "EXPIRED_API_KEY":
                        st.error("This key has expired. Generate a new one.")
                    elif code == "REVOKED_API_KEY":
                        st.error("This key has been revoked. Generate a new one.")
                    else:
                        st.error("Key not recognised.")
                else:
                    st.error("Key rejected.")
    st.stop()


# ═════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 🔍 Predict
# ─────────────────────────────────────────────────────────────
if page == "🔍 Predict":
    st.title("🔍 Predict Customer Churn")

    # Fetch allowed models once
    mr = _req("GET", "/v1/models", headers=_ah())
    if mr.status_code == 401:
        st.error("Your API key has expired or been revoked. Go to **Manage Keys**.")
        st.stop()
    mdata   = _json(mr)
    allowed = mdata.get("your_allowed_models", [])
    amap    = {m["model_id"]: m["description"] for m in mdata.get("available_models", [])}

    if not allowed:
        st.warning("Your role has no model access. Ask an admin to upgrade your role.")
        st.stop()

    # ── All inputs inside a form — NO reruns until Predict is clicked ──────
    with st.form("predict_form"):
        st.subheader("Customer Features")
        c1, c2, c3 = st.columns(3)

        with c1:
            age      = st.slider("Age", 18, 100, 35)
            tenure   = st.slider("Tenure (months)", 0, 120, 24)
            usage    = st.slider("Usage Frequency", 0, 30, 10)
            calls    = st.slider("Support Calls", 0, 20, 2)

        with c2:
            delay    = st.slider("Payment Delay (days)", 0, 30, 5)
            last_int = st.slider("Last Interaction (days ago)", 0, 365, 30)
            spend    = st.number_input("Total Spend ($)", 0.0, 10000.0, 500.0, step=50.0)

        with c3:
            gender   = st.selectbox("Gender", ["Male", "Female"])
            sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            chosen   = st.selectbox(
                "Model",
                allowed,
                format_func=lambda x: f"{x} — {amap.get(x, x)}",
            )

        submitted = st.form_submit_button(
            "🚀 Run Prediction",
            type="primary",
            use_container_width=True,
        )

    # ── Only runs when button is clicked — sliders are silent ─────────────
    if submitted:
        payload = {
            "model_id": chosen, "Age": age, "Gender": gender,
            "Tenure": tenure, "Usage Frequency": usage,
            "Support Calls": calls, "Payment Delay": delay,
            "Subscription Type": sub_type, "Contract Length": contract,
            "Total Spend": spend, "Last Interaction": last_int,
        }
        with st.spinner("Running prediction…"):
            pr     = _req("POST", "/v1/predict", json=payload, headers=_ah())
        result = _json(pr)

        if pr.status_code == 200:
            prob = result["churn_probability"]
            pred = result["churn_prediction"]
            # Store result so it persists across sidebar navigation
            st.session_state.pred_result = result
            st.session_state.pred_inputs = {
                "Age": age, "Gender": gender, "Tenure": tenure,
                "Subscription Type": sub_type, "Model": chosen,
            }
        else:
            st.error(_detail(pr))
            st.session_state.pred_result = None

    # ── Show last result (survives slider interactions) ────────────────────
    if st.session_state.pred_result:
        r      = st.session_state.pred_result
        prob   = r["churn_probability"]
        pred   = r["churn_prediction"]

        st.divider()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            if pred == 1:
                st.error("⚠️ **Likely to Churn**")
            else:
                st.success("✅ **Likely to Stay**")
            st.metric("Churn Probability", f"{prob:.1%}")
            st.progress(prob)
            st.caption(f"Model: `{r['model_id']}`")
        with col_b:
            st.markdown("**Inputs used for this prediction**")
            for k, v in st.session_state.pred_inputs.items():
                st.write(f"- **{k}:** {v}")


# ─────────────────────────────────────────────────────────────
# 🔄 Retrain
# ─────────────────────────────────────────────────────────────
elif page == "🔄 Retrain":
    st.title("🔄 Model Retraining")

    if st.session_state.role not in {"admin", "analyst"}:
        st.error(f"Requires analyst or admin role. Yours: `{st.session_state.role}`")
        st.stop()

    st.info("Upload a CSV with feature columns + a `Churn` column (0 or 1).")

    mr      = _req("GET", "/v1/models", headers=_ah())
    allowed = _json(mr).get("your_allowed_models", []) if mr.status_code == 200 else []
    model_id = st.selectbox("Model to retrain", allowed or ["rf"])

    uploaded = st.file_uploader("Training CSV", type=["csv"])
    if uploaded:
        import pandas as pd
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"{len(df)} rows · {len(df.columns)} columns")

        if "Churn" not in df.columns:
            st.error("CSV must have a `Churn` column.")
        else:
            if st.button("🚀 Submit Retrain Job", type="primary"):
                payload = {
                    "model_id": model_id,
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
    st.subheader("Poll Job Status")
    jid = st.text_input("Job ID", value=st.session_state.last_job_id)
    if st.button("🔄 Refresh Status") and jid:
        r = _req("GET", f"/v1/retrain/{jid}", headers=_ah())
        d = _json(r)
        if r.status_code == 200:
            s    = d["status"]
            icon = {"queued": "🟡", "running": "🔵", "done": "🟢", "failed": "🔴"}.get(s, "⚪")
            st.write(f"**Status:** {icon} {s.title()}")
            st.write(f"Model: `{d['model_id']}` · Rows: {d['rows_received']}")
            if d.get("metrics"):
                st.json(d["metrics"])
            if d.get("error"):
                st.error(d["error"])
        else:
            st.error(_detail(r))

    if st.session_state.role == "admin":
        st.divider()
        if st.button("Load All Job History"):
            r = _req("GET", "/v1/retrain", headers=_ah())
            if r.status_code == 200:
                import pandas as pd
                jdf = pd.DataFrame(_json(r))
                if not jdf.empty:
                    st.dataframe(
                        jdf[["job_id", "model_id", "status", "rows_received",
                             "triggered_by", "created_at"]],
                        use_container_width=True,
                    )


# ─────────────────────────────────────────────────────────────
# 📊 Models
# ─────────────────────────────────────────────────────────────
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
    else:
        st.error(_detail(mr))

    st.divider()
    rr = _req("GET", "/v1/auth/roles")
    if rr and rr.status_code == 200:
        st.subheader("Role permissions")
        for role, models in _json(rr).items():
            colour = ROLE_COLOUR.get(role, "#888")
            st.markdown(
                f"<span style='color:{colour};font-weight:700'>{role.title()}</span>: "
                + " · ".join(f"`{m}`" for m in models),
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────
# 🔑 Manage Keys  (own keys only — no other user's data visible)
# ─────────────────────────────────────────────────────────────
elif page == "🔑 Manage Keys":
    st.title("🔑 Manage Your API Keys")

    st.write(f"**Active session key prefix:** `{st.session_state.key_prefix}`")
    if st.session_state.key_expires_at:
        st.write(f"**Expires:** {st.session_state.key_expires_at[:10]}")

    st.divider()
    col_new, col_rev = st.columns(2)

    with col_new:
        st.subheader("Generate a new key")
        st.caption("Issues a fresh key. Your current key stays active until it expires or is revoked.")
        with st.form("new_key_form"):
            np_ = st.text_input("Confirm password", type="password")
            go  = st.form_submit_button("Generate New Key", type="primary",
                                        use_container_width=True)
        if go:
            r    = _req("POST", "/v1/auth/keys",
                        json={"username": st.session_state.username, "password": np_})
            data = _json(r)
            if r.status_code == 200:
                st.success("✅ New key generated — copy it now:")
                st.code(data["raw_key"], language=None)
                st.caption(f"Expires: {data['expires_at'][:10]}")
                st.session_state.api_key        = data["raw_key"]
                st.session_state.key_prefix     = data["prefix"]
                st.session_state.key_expires_at = data["expires_at"]
            else:
                st.error(_detail(r))

    with col_rev:
        st.subheader("Revoke a key")
        st.caption("Enter the prefix shown at key creation (e.g. `sk-abc12`).")
        with st.form("revoke_form"):
            rp_  = st.text_input("Confirm password", type="password")
            rpfx = st.text_input("Key prefix to revoke")
            rev  = st.form_submit_button("Revoke Key", type="primary",
                                         use_container_width=True)
        if rev:
            r    = _req("DELETE", f"/v1/auth/keys/{rpfx}",
                        json={"username": st.session_state.username, "password": rp_})
            data = _json(r)
            if r.status_code == 200:
                st.success(data.get("message", "Revoked."))
                if rpfx.startswith(st.session_state.key_prefix):
                    st.warning("You revoked your active session key — logging out.")
                    for k in DEFAULTS:
                        st.session_state[k] = DEFAULTS[k]
                    st.rerun()
            else:
                st.error(_detail(r))


# ─────────────────────────────────────────────────────────────
# 🛡️ Admin Panel
# ─────────────────────────────────────────────────────────────
elif page == "🛡️ Admin Panel":
    st.title("🛡️ Admin Panel")

    # ── Secondary auth — admin must re-confirm password ───────
    # This means even if someone steals a session cookie they still
    # can't promote themselves without knowing the admin password.
    if "admin_confirmed" not in st.session_state:
        st.session_state.admin_confirmed = False

    if not st.session_state.admin_confirmed:
        st.warning("Admin panel requires password confirmation.")
        with st.form("admin_confirm"):
            ap = st.text_input("Admin password", type="password")
            ok = st.form_submit_button("Confirm", type="primary")
        if ok:
            r = _req("POST", "/v1/auth/login",
                     json={"username": st.session_state.username, "password": ap})
            if r.status_code == 200 and _json(r).get("role") == "admin":
                st.session_state.admin_confirmed = True
                st.session_state.admin_password  = ap  # kept for API calls below
                st.rerun()
            else:
                st.error("Wrong password or insufficient role.")
        st.stop()

    # ── Confirmed admin from here ──────────────────────────────
    tab_users, tab_roles, tab_newuser = st.tabs(
        ["👥 Users", "🔐 Change Role", "➕ Create User"]
    )

    # ── Users table ───────────────────────────────────────────
    with tab_users:
        st.subheader("All Users")
        if st.button("🔄 Refresh", key="refresh_users"):
            st.session_state.pop("admin_users_cache", None)

        if "admin_users_cache" not in st.session_state:
            r = _req(
                "GET", "/v1/auth/users",
                params={
                    "admin_username": st.session_state.username,
                    "admin_password": st.session_state.admin_password,
                },
            )
            st.session_state.admin_users_cache = _json(r) if r.status_code == 200 else []

        users = st.session_state.admin_users_cache
        if users:
            import pandas as pd
            udf = pd.DataFrame(users)[["id", "username", "role", "is_active", "created_at"]]
            # Colour-code roles
            def _colour_role(val):
                c = ROLE_COLOUR.get(val, "#888")
                return f"color: {c}; font-weight: bold"
            st.dataframe(
                udf.style.applymap(_colour_role, subset=["role"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No users found or failed to load.")

    # ── Change role ───────────────────────────────────────────
    with tab_roles:
        st.subheader("Change a user's role")
        st.caption(
            "Role controls which models a user can access and whether they can retrain.  \n"
            "**guest** → logreg only · **analyst** → rf, gb, xgb + retrain · "
            "**admin** → all models + retrain + admin panel"
        )

        users = st.session_state.get("admin_users_cache", [])
        usernames = [u["username"] for u in users if u["username"] != st.session_state.username]

        if not usernames:
            st.info("Load the Users tab first to populate the list.")
        else:
            with st.form("role_form"):
                target   = st.selectbox("Select user", usernames)
                new_role = st.selectbox("New role", ["guest", "analyst", "admin"])

                current_role = next(
                    (u["role"] for u in users if u["username"] == target), "unknown"
                )
                st.caption(f"Current role: **{_badge(current_role)}**")

                go = st.form_submit_button("Update Role", type="primary",
                                           use_container_width=True)

            if go:
                r = _req(
                    "PATCH", f"/v1/auth/users/{target}/role",
                    params={
                        "admin_username": st.session_state.username,
                        "admin_password": st.session_state.admin_password,
                    },
                    json={"role": new_role},
                )
                if r.status_code == 200:
                    st.success(
                        f"✅ **{target}** is now `{new_role}`. "
                        "They need to generate a new key for the new TTL to apply."
                    )
                    # Invalidate cache
                    st.session_state.pop("admin_users_cache", None)
                else:
                    st.error(_detail(r))

        st.divider()
        st.subheader("Enable / Disable a user")
        if not usernames:
            st.info("Load the Users tab first.")
        else:
            with st.form("toggle_form"):
                t2     = st.selectbox("Select user", usernames, key="toggle_user")
                active = st.toggle("Account active", value=True)
                go2    = st.form_submit_button("Apply", type="primary",
                                               use_container_width=True)
            if go2:
                r = _req(
                    "PATCH", f"/v1/auth/users/{t2}/activate",
                    params={
                        "admin_username": st.session_state.username,
                        "admin_password": st.session_state.admin_password,
                        "active": str(active).lower(),
                    },
                )
                if r.status_code == 200:
                    state = "enabled" if active else "disabled"
                    st.success(f"✅ Account **{t2}** {state}.")
                    st.session_state.pop("admin_users_cache", None)
                else:
                    st.error(_detail(r))

    # ── Create user (admin creating accounts for others) ──────
    with tab_newuser:
        st.subheader("Create a user account")
        st.caption(
            "Use this to provision analyst or admin accounts directly — "
            "no need for them to sign up through the public form."
        )
        with st.form("admin_create_user"):
            nu   = st.text_input("Username")
            np2  = st.text_input("Temporary password", type="password",
                                 help="User should change this after first login.")
            nr   = st.selectbox("Role", ["guest", "analyst", "admin"])
            go3  = st.form_submit_button("Create Account", type="primary",
                                         use_container_width=True)

        if go3:
            if not nu or not np2:
                st.warning("Fill in all fields.")
            else:
                # Create the account
                r = _req("POST", "/v1/auth/signup", json={"username": nu, "password": np2})
                data = _json(r)
                if r.status_code == 201:
                    # Immediately promote to the requested role if not guest
                    if nr != "guest":
                        r2 = _req(
                            "PATCH", f"/v1/auth/users/{nu}/role",
                            params={
                                "admin_username": st.session_state.username,
                                "admin_password": st.session_state.admin_password,
                            },
                            json={"role": nr},
                        )
                        if r2.status_code != 200:
                            st.warning(
                                f"Account created but role promotion failed: {_detail(r2)}"
                            )
                            st.stop()

                    st.success(
                        f"✅ Account **{nu}** created with role `{nr}`.  \n"
                        f"Share these credentials securely:  \n"
                        f"- Username: `{nu}`  \n"
                        f"- Password: `{np2}` *(temporary)*"
                    )
                    st.session_state.pop("admin_users_cache", None)
                elif r.status_code == 409:
                    st.error("Username already taken.")
                elif r.status_code == 422:
                    for d in data.get("details", []):
                        st.error(f"**{d.get('field')}**: {d.get('message')}")
                else:
                    st.error(_detail(r))