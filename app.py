from pathlib import Path

import streamlit as st

from src.retriever.fashion_context_retriever import (
    FashionContextRetriever,
    parse_query,
)

PROJECT_ROOT = Path(__file__).resolve().parent

TOP_K = 5

EXAMPLE_QUERIES = [
    ("Yellow raincoat", "A person in a bright yellow raincoat"),
    ("Park bench", "Someone wearing a blue shirt sitting on a park bench"),
    ("City walk", "Casual weekend outfit for a city walk"),
    ("Modern office", "Professional business attire inside a modern office"),
    ("Red tie + white shirt", "A red tie and a white shirt in a formal setting"),
]


st.set_page_config(
    page_title="Glance ML Assignment - Fashion Retrieval",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #faf8f4;
        color: #16130f;
    }

    header {
        background: transparent !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .brand-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 4.5rem;
    }

    .brand {
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.34rem;
        text-transform: uppercase;
        color: #16130f;
    }

    .brand-tag {
        font-size: 0.72rem;
        letter-spacing: 0.16rem;
        text-transform: uppercase;
        color: #8d8578;
    }

    .eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.24rem;
        text-transform: uppercase;
        color: #6a4cff;
        font-weight: 700;
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-size: clamp(3.2rem, 6.5vw, 6.2rem);
        line-height: 0.95;
        letter-spacing: -0.3rem;
        font-weight: 800;
        color: #16130f;
        margin-bottom: 1.6rem;
    }

    .hero-title em {
        font-style: normal;
        color: #6a4cff;
    }

    .hero-copy {
        max-width: 560px;
        color: #6f6759;
        font-size: 1.08rem;
        line-height: 1.65;
        margin-bottom: 2.4rem;
    }

    div[data-testid="stForm"] {
        border: 0;
        padding: 0;
    }

    div[data-testid="stTextInput"] input {
        background: #ffffff;
        color: #16130f;
        border: 1.5px solid #e2dccf;
        border-radius: 16px;
        padding: 1.15rem 1.3rem;
        font-family: inherit;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.015rem;
        box-shadow: 0 2px 14px rgba(22, 19, 15, 0.04);
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #6a4cff;
        box-shadow: 0 0 0 1px #6a4cff;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #a89f8f;
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3.55rem;
        border: 0;
        border-radius: 16px;
        background: #16130f;
        color: #faf8f4;
        font-weight: 700;
        letter-spacing: 0.03rem;
        font-size: 0.98rem;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #6a4cff;
        color: #ffffff;
    }

    .try-label {
        font-size: 0.72rem;
        letter-spacing: 0.2rem;
        text-transform: uppercase;
        color: #8d8578;
        margin: 1.6rem 0 0.6rem 0;
    }

    .stButton > button {
        border: 1.5px solid #e2dccf;
        border-radius: 999px;
        background: #ffffff;
        color: #4b443a;
        font-size: 0.86rem;
        font-weight: 600;
        padding: 0.42rem 1.05rem;
        white-space: nowrap;
    }

    .stButton > button:hover {
        border-color: #6a4cff;
        color: #6a4cff;
        background: #ffffff;
    }

    .section-space {
        height: 3.4rem;
    }

    .section-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.1rem;
        color: #16130f;
        margin-bottom: 0.4rem;
    }

    .section-copy {
        color: #8d8578;
        margin-bottom: 1.8rem;
        font-size: 0.98rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 8px 28px rgba(22, 19, 15, 0.10);
    }

    .result-rank {
        color: #b3a995;
        font-size: 0.68rem;
        letter-spacing: 0.18rem;
        font-weight: 700;
        margin-top: 0.85rem;
    }

    .result-score {
        color: #55503f;
        font-size: 0.87rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1.5px solid #eee8db;
        border-radius: 14px;
    }

    div[data-testid="stExpander"] details {
        background: #ffffff;
        border-radius: 14px;
    }

    div[data-testid="stExpander"] summary {
        background: #ffffff;
        color: #16130f !important;
        border-radius: 14px;
    }

    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {
        color: #16130f !important;
        font-weight: 600;
        font-size: 0.88rem;
    }

    div[data-testid="stExpander"] summary svg {
        fill: #6a4cff !important;
        color: #6a4cff !important;
    }

    div[data-testid="stExpander"] summary:hover p {
        color: #6a4cff !important;
    }

    .metric-name {
        color: #8d8578;
        font-size: 0.9rem;
    }

    .metric-value {
        float: right;
        color: #16130f;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .error-card {
        background: #ffffff;
        border: 1.5px solid #e8c8c2;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        max-width: 640px;
    }

    .error-title {
        font-weight: 800;
        font-size: 1.15rem;
        color: #16130f;
        margin-bottom: 0.5rem;
    }

    .error-copy {
        color: #6f6759;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .app-footer {
        margin-top: 4.5rem;
        padding-top: 1.4rem;
        border-top: 1.5px solid #eee8db;
        color: #a89f8f;
        font-size: 0.8rem;
        letter-spacing: 0.03rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_retriever():
    return FashionContextRetriever()


def queue_search(query_text):
    st.session_state.query_input = query_text
    st.session_state.pending_search = True


def format_fashion(parsed):
    clauses = [clause["text"] for clause in parsed["fashion_clauses"]]

    if not clauses:
        return "No explicit garment constraint"

    return " · ".join(clauses)


def render_interpretation(parsed):
    with st.expander("How we interpreted your search", expanded=False):
        rows = [
            ("Garment constraints", format_fashion(parsed)),
            ("Style", parsed["style"] or "Open style"),
            ("Setting", parsed["context"] or "Open setting"),
        ]

        body = "<br>".join(
            f'<span class="metric-name">{label}</span>'
            f'<span class="metric-value">{value}</span>'
            for label, value in rows
        )

        st.markdown(body, unsafe_allow_html=True)


def render_result(result):
    image_path = PROJECT_ROOT / result["image_path"]

    st.image(
        str(image_path),
        width="stretch",
    )

    st.markdown(
        f"""
        <div class="result-rank">LOOK {int(result["rank"]):02d}</div>
        <div class="result-score">
            Relative retrieval score {float(result["final_score"]):.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why this look?"):
        rows = [
            ("Garment match", f"{result['global_score']:.3f}"),
            ("Region evidence", f"{result['region_score']:.3f}"),
            ("Constraint coverage", f"{result['coverage_score']:.2f}"),
            ("Style alignment", f"{result['style_score']:.3f}"),
            ("Scene context", f"{result['context_score']:.3f}"),
        ]

        body = "<br>".join(
            f'<span class="metric-name">{name}</span>'
            f'<span class="metric-value">{value}</span>'
            for name, value in rows
        )

        st.markdown(body, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Header + hero
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="brand-row">
        <div class="brand">Glance</div>
        <div class="brand-tag">ML Assignment Demo</div>
    </div>

    <div class="eyebrow">ML Internship Assignment</div>

    <div class="hero-title">
        Find the look<br><em>you have in mind.</em>
    </div>

    <div class="hero-copy">
        Describe the outfit, style, and setting. The retrieval system grounds
        garment constraints and scene context to find relevant looks.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Search: form (Enter + button share one path) and example chips
# --------------------------------------------------------------------------

if "pending_search" not in st.session_state:
    st.session_state.pending_search = False

with st.form("search_form", clear_on_submit=False):
    input_column, button_column = st.columns([5, 1])

    with input_column:
        st.text_input(
            "Search",
            key="query_input",
            label_visibility="collapsed",
            placeholder="Describe a garment, colour, style and setting...",
        )

    with button_column:
        submitted = st.form_submit_button("Discover", width="stretch")

if submitted:
    st.session_state.pending_search = True

st.markdown('<div class="try-label">Try a search</div>', unsafe_allow_html=True)

chip_columns = st.columns(len(EXAMPLE_QUERIES))

for column, (chip_label, full_query) in zip(chip_columns, EXAMPLE_QUERIES):
    with column:
        st.button(
            chip_label,
            key=f"chip_{chip_label}",
            on_click=queue_search,
            args=(full_query,),
            width="stretch",
        )


# --------------------------------------------------------------------------
# Search execution (single path for button, Enter, and chips)
# --------------------------------------------------------------------------

if st.session_state.pending_search:
    st.session_state.pending_search = False

    query = st.session_state.get("query_input", "").strip()

    if not query:
        st.warning("Describe the look you want to find.")
    else:
        try:
            retriever = load_retriever()
        except Exception as error:
            st.markdown(
                """
                <div class="error-card">
                    <div class="error-title">Retrieval artifacts unavailable</div>
                    <div class="error-copy">
                        The local embeddings and indexes this demo needs were not
                        found. Build them first (see the README data preparation
                        steps), then reload this page.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")

            st.stop()

        with st.spinner("Curating looks..."):
            st.session_state.results = retriever.search(query=query, top_k=TOP_K)
            st.session_state.parsed = parse_query(query)
            st.session_state.last_query = query


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

if "results" in st.session_state:
    results = st.session_state.results
    parsed = st.session_state.parsed

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="section-title">Your closest looks.</div>
        <div class="section-copy">
            Ranked by garment evidence, style, and scene context from
            &ldquo;{st.session_state.last_query}&rdquo;.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_interpretation(parsed)

    st.markdown("<div style='height: 1.4rem'></div>", unsafe_allow_html=True)

    result_rows = list(results.iterrows())

    lead_row = st.columns([1.7, 1, 1])

    for column, (_, result) in zip(lead_row, result_rows[:3]):
        with column:
            render_result(result)

    if len(result_rows) > 3:
        st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)

        second_row = st.columns(3)

        for column, (_, result) in zip(second_row, result_rows[3:]):
            with column:
                render_result(result)


# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-footer">
        ML internship assignment demo · Built for the Glance ML assignment ·
        Not an official Glance product
    </div>
    """,
    unsafe_allow_html=True,
)
