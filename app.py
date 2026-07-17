from pathlib import Path

import streamlit as st

from src.retriever.fashion_context_retriever import (
    FashionContextRetriever,
    parse_query,
)

PROJECT_ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Contextual Fashion",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 80% 10%,
                rgba(255, 80, 120, 0.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 10% 40%,
                rgba(126, 87, 255, 0.12),
                transparent 30%
            ),
            #080808;
        color: #f5f5f5;
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
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .brand {
        font-size: 0.8rem;
        letter-spacing: 0.28rem;
        font-weight: 700;
        color: #f4f4f4;
        margin-bottom: 5rem;
    }

    .spark {
        color: #ff6d9f;
    }

    .eyebrow {
        color: #a7a7a7;
        font-size: 0.75rem;
        letter-spacing: 0.2rem;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(3.5rem, 7vw, 7.5rem);
        line-height: 0.88;
        letter-spacing: -0.42rem;
        font-weight: 800;
        max-width: 1050px;
        margin-bottom: 2rem;
    }

    .hero-gradient {
        background:
            linear-gradient(
                90deg,
                #ffffff 0%,
                #ff8db4 48%,
                #9f8cff 100%
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-copy {
        max-width: 660px;
        color: #a9a9a9;
        font-size: 1.15rem;
        line-height: 1.7;
        margin-bottom: 2.5rem;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(255, 255, 255, 0.07);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        font-size: 1rem;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #ff78a8;
        box-shadow: 0 0 0 1px #ff78a8;
    }

    .stButton > button {
        width: 100%;
        min-height: 3.4rem;
        border: 0;
        border-radius: 18px;
        background:
            linear-gradient(
                100deg,
                #ff6f9f,
                #9b7cff
            );
        color: white;
        font-weight: 700;
        letter-spacing: 0.04rem;
    }

    .stButton > button:hover {
        color: white;
        border: 0;
        transform: translateY(-1px);
    }

    .section-space {
        height: 5rem;
    }

    .section-title {
        font-size: 2.6rem;
        font-weight: 750;
        letter-spacing: -0.12rem;
        margin-bottom: 0.5rem;
    }

    .section-copy {
        color: #8f8f8f;
        margin-bottom: 2rem;
    }

    .intent-card {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        min-height: 125px;
    }

    .intent-label {
        color: #8b8b8b;
        font-size: 0.68rem;
        letter-spacing: 0.14rem;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .intent-value {
        color: #ffffff;
        font-size: 1.05rem;
        line-height: 1.5;
    }

    div[data-testid="stImage"] img {
        border-radius: 24px;
    }

    .rank {
        color: #ff7eaa;
        font-size: 0.72rem;
        letter-spacing: 0.14rem;
        font-weight: 700;
        margin-top: 0.8rem;
    }

    .image-id {
        font-size: 1.05rem;
        font-weight: 650;
        margin-top: 0.25rem;
    }

    .match {
        color: #8f8f8f;
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }

    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
    }

    .metric-name {
        color: #929292;
    }

    .metric-value {
        float: right;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_retriever():
    return FashionContextRetriever()


def format_fashion(parsed):
    clauses = [clause["text"] for clause in parsed["fashion_clauses"]]

    if not clauses:
        return "No explicit garment constraint"

    return " · ".join(clauses)


def display_intent(parsed):
    columns = st.columns(3)

    values = [
        (
            "Fashion",
            format_fashion(parsed),
        ),
        (
            "Style",
            parsed["style"] or "Open style",
        ),
        (
            "Context",
            parsed["context"] or "Open setting",
        ),
    ]

    for column, (
        label,
        value,
    ) in zip(columns, values):
        with column:
            st.markdown(
                f"""
                <div class="intent-card">
                    <div class="intent-label">
                        {label}
                    </div>
                    <div class="intent-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_result(result):
    image_path = PROJECT_ROOT / result["image_path"]

    st.image(
        str(image_path),
        use_container_width=True,
    )

    match_percentage = max(
        0,
        min(
            100,
            round(float(result["final_score"]) * 100),
        ),
    )

    st.markdown(
        f"""
        <div class="rank">
            MATCH #{int(result["rank"]):02d}
        </div>

        <div class="image-id">
            {result["image_id"]}
        </div>

        <div class="match">
            {match_percentage}% retrieval match
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why this look?"):
        st.markdown(
            f"""
            <span class="metric-name">
                Global fashion
            </span>
            <span class="metric-value">
                {result["global_score"]:.3f}
            </span>
            <br>

            <span class="metric-name">
                Region match
            </span>
            <span class="metric-value">
                {result["region_score"]:.3f}
            </span>
            <br>

            <span class="metric-name">
                Clause coverage
            </span>
            <span class="metric-value">
                {result["coverage_score"]:.2f}
            </span>
            <br>

            <span class="metric-name">
                Style
            </span>
            <span class="metric-value">
                {result["style_score"]:.3f}
            </span>
            <br>

            <span class="metric-name">
                Context
            </span>
            <span class="metric-value">
                {result["context_score"]:.3f}
            </span>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class="brand">
        <span class="spark">✦</span>
        CONTEXTUAL FASHION
    </div>

    <div class="eyebrow">
        Multimodal fashion discovery
    </div>

    <div class="hero-title">
        Find the look<br>
        <span class="hero-gradient">
            you're imagining.
        </span>
    </div>

    <div class="hero-copy">
        Search across clothing, colour, style and
        environment. Describe the moment — the retrieval
        engine binds fashion details to the scene around them.
    </div>
    """,
    unsafe_allow_html=True,
)


example_queries = [
    "A person in a bright yellow raincoat",
    "Someone wearing a blue shirt sitting on a park bench",
    "Casual weekend outfit for a city walk",
    "Professional business attire inside a modern office",
    "A red tie and a white shirt in a formal setting",
]


if "query" not in st.session_state:
    st.session_state.query = "Someone wearing a blue shirt sitting on a park bench"


search_column, button_column = st.columns([5, 1])

with search_column:
    query = st.text_input(
        "Search",
        key="query",
        label_visibility="collapsed",
        placeholder=("Describe a garment, colour, style and setting..."),
    )

with button_column:
    search_clicked = st.button(
        "DISCOVER ✦",
        use_container_width=True,
    )


st.markdown(
    "<div style='height: 1rem'></div>",
    unsafe_allow_html=True,
)

st.caption(
    "Try: yellow raincoat · business attire in an office "
    "· blue shirt on a park bench · casual city walk"
)


if search_clicked:
    if not query.strip():
        st.warning("Describe the look you want to discover.")

    else:
        retriever = load_retriever()

        with st.spinner("Understanding your vibe..."):
            results = retriever.search(
                query=query,
                top_k=5,
            )

            parsed = parse_query(query)

        st.markdown(
            '<div class="section-space"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="eyebrow">
                Query intelligence
            </div>

            <div class="section-title">
                We understood your vibe.
            </div>

            <div class="section-copy">
                Fashion, style and environment are scored
                as separate signals before final reranking.
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_intent(parsed)

        st.markdown(
            '<div class="section-space"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="eyebrow">
                Curated by retrieval
            </div>

            <div class="section-title">
                Your closest matches.
            </div>

            <div class="section-copy">
                Ranked using global fashion similarity,
                garment-region evidence and scene context.
            </div>
            """,
            unsafe_allow_html=True,
        )

        first_row = st.columns([1.15, 1, 1])

        for column, (
            _,
            result,
        ) in zip(
            first_row,
            results.iloc[:3].iterrows(),
        ):
            with column:
                display_result(result)

        second_row = st.columns([1, 1, 1])

        for column, (
            _,
            result,
        ) in zip(
            second_row,
            results.iloc[3:].iterrows(),
        ):
            with column:
                display_result(result)
