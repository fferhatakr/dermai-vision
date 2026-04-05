import streamlit as st

def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Base reset ── */
    *, *::before, *::after { box-sizing: border-box; }

    html, body, .stApp {
        background-color: #080c12 !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #c8d0dc !important;
    }

    .block-container {
        padding: 2rem 2.5rem 4rem !important;
        max-width: 1200px !important;
    }

    /* ── Header bar ── */
    .ds-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #1a2235;
        padding-bottom: 1.25rem;
        margin-bottom: 2rem;
    }
    .ds-logo {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        color: #3b82f6;
        text-transform: uppercase;
    }
    .ds-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #e2e8f0;
        margin: 0;
    }
    .ds-user-badge {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #64748b;
        letter-spacing: 0.05em;
    }

    /* ── Section labels ── */
    .ds-section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.75rem;
    }

    /* ── Cards ── */
    .ds-card {
        background: #0d1420;
        border: 1px solid #1a2235;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .ds-card-accent {
        border-left: 3px solid #3b82f6;
    }

    /* ── Mode selector ── */
    .ds-mode-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .ds-mode-card {
        background: #0d1420;
        border: 1px solid #1a2235;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        cursor: pointer;
        transition: border-color 0.15s, background 0.15s;
    }
    .ds-mode-card:hover { border-color: #3b82f6; }
    .ds-mode-card.active {
        border-color: #3b82f6;
        background: #0f1e35;
    }
    .ds-mode-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #e2e8f0;
        margin-bottom: 0.2rem;
    }
    .ds-mode-desc {
        font-size: 0.72rem;
        color: #64748b;
        line-height: 1.5;
    }
    .ds-mode-tag {
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        margin-top: 0.5rem;
    }
    .tag-fast { background: #0f3d2e; color: #34d399; }
    .tag-balanced { background: #1a2f4a; color: #60a5fa; }
    .tag-full { background: #2d1f4a; color: #a78bfa; }

    /* ── Result blocks ── */
    .ds-result-risk {
        background: #1a0f0f;
        border: 1px solid #7f1d1d;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .ds-result-safe {
        background: #0a1f14;
        border: 1px solid #14532d;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .ds-result-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .ds-result-label.risk { color: #f87171; }
    .ds-result-label.safe { color: #4ade80; }
    .ds-result-diagnosis {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: -0.01em;
    }
    .ds-result-conf {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* ── Probability bars ── */
    .ds-prob-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.55rem;
    }
    .ds-prob-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: #94a3b8;
        width: 48px;
        flex-shrink: 0;
    }
    .ds-prob-bar-bg {
        flex: 1;
        height: 5px;
        background: #1a2235;
        border-radius: 3px;
        overflow: hidden;
    }
    .ds-prob-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: #3b82f6;
        transition: width 0.4s ease;
    }
    .ds-prob-bar-fill.top { background: #f87171; }
    .ds-prob-value {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: #64748b;
        width: 38px;
        text-align: right;
        flex-shrink: 0;
    }

    /* ── Conflict / consensus banners ── */
    .ds-conflict-banner {
        background: #1c1208;
        border: 1px solid #78350f;
        border-radius: 7px;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: #fbbf24;
        margin-bottom: 1rem;
    }
    .ds-consensus-banner {
        background: #0a1a10;
        border: 1px solid #166534;
        border-radius: 7px;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: #86efac;
        margin-bottom: 1rem;
    }

    /* ── Warning / info strips ── */
    .ds-warning-strip {
        background: #1a1206;
        border-left: 3px solid #d97706;
        padding: 0.65rem 1rem;
        font-size: 0.78rem;
        color: #fbbf24;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.75rem;
    }
    .ds-info-strip {
        background: #0c1829;
        border-left: 3px solid #1d4ed8;
        padding: 0.65rem 1rem;
        font-size: 0.78rem;
        color: #93c5fd;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.75rem;
    }

    /* ── Debug panel ── */
    .ds-debug-panel {
        background: #080c12;
        border: 1px solid #1a2235;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
    }
    .ds-debug-title {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.75rem;
    }
    .ds-debug-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        border-bottom: 1px solid #0f1825;
        font-size: 0.75rem;
    }
    .ds-debug-row:last-child { border-bottom: none; }
    .ds-debug-key { color: #64748b; }
    .ds-debug-val {
        font-family: 'DM Mono', monospace;
        color: #94a3b8;
    }

    /* ── Image containers ── */
    .ds-img-wrap {
        border: 1px solid #1a2235;
        border-radius: 8px;
        overflow: hidden;
    }
    .ds-img-caption {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #475569;
        padding: 0.5rem 0.75rem;
        border-top: 1px solid #1a2235;
    }

    /* ── Divider ── */
    .ds-divider {
        border: none;
        border-top: 1px solid #1a2235;
        margin: 1.75rem 0;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: #1d4ed8 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 7px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.03em !important;
        padding: 0.6rem 1.5rem !important;
        transition: background 0.15s !important;
    }
    .stButton > button:hover { background: #1e40af !important; }

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: #0d1420 !important;
        border: 1px solid #1a2235 !important;
        border-radius: 7px !important;
        color: #c8d0dc !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stFileUploader > label,
    .stRadio > label {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0d1420 !important;
        border: 1px dashed #1a2235 !important;
        border-radius: 8px !important;
    }

    .stProgress > div > div > div > div {
        background: #3b82f6 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1a2235 !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    div[data-testid="stAlert"] {
        background: #0d1420 !important;
        border: 1px solid #1a2235 !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #3b82f6 !important; }

    /* ── Mobile responsive ── */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 1.1rem 3rem !important; }
        .ds-mode-grid { grid-template-columns: 1fr; }
        .ds-header { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
        .ds-result-diagnosis { font-size: 1.15rem; }
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .stButton > button { width: 100% !important; }
    }
    @media (max-width: 480px) {
        .ds-title { font-size: 0.95rem; }
    }
    </style>
    """, unsafe_allow_html=True)