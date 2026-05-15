import streamlit as st
import pickle
import numpy as np
import requests
import os
import re

# ─────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "a16dae945050d925b1bd757c045394a2")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMG = "https://via.placeholder.com/500x750/1a1a2e/ffffff?text=No+Image"

# ─────────────────────────────────────────
#  LOAD DATA  (cached so it only runs once)
# ─────────────────────────────────────────
@st.cache_resource
def load_data():
    movies = pickle.load(open("movies_df.pkl", "rb"))
    sim    = np.load("similarity.npy")
    return movies, sim

movies_df, similarity = load_data()

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────

def format_cast(cast_list):
    """Turn a raw cast list into a readable string like 'Sam Worthington, Zoe Saldana'."""
    if not isinstance(cast_list, list) or len(cast_list) == 0:
        return "Cast unavailable"

    formatted = []
    for name in cast_list[:3]:                         # show top 3 actors
        # raw names are like 'samworthington' → add spaces before capitals
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        formatted.append(name.title())

    return ", ".join(formatted)


def format_overview(overview):
    """Turn overview (list or string) into a plain paragraph."""
    if isinstance(overview, list):
        return " ".join(overview)
    return str(overview) if overview else "No overview available."


def format_genres(genres):
    """Return a list of title-cased genre strings."""
    if isinstance(genres, list):
        return [g.replace("sciencefiction", "Sci-Fi").title() for g in genres]
    return []


def fetch_poster(movie_title):
    """Fetch the poster URL from TMDB. Returns placeholder if not found."""
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": movie_title},
            timeout=5,
        )
        results = response.json().get("results", [])
        if results and results[0].get("poster_path"):
            return TMDB_IMG_BASE + results[0]["poster_path"]
    except Exception:
        pass  # silently fall back to placeholder
    return PLACEHOLDER_IMG


def get_recommendations(selected_title, num_results=6):
    """Return a list of (movie_row, similarity_score) for the top matches."""
    # Find the row index for the selected movie
    match = movies_df[movies_df["title"] == selected_title]
    if match.empty:
        return []

    idx    = match.index[0]
    scores = list(enumerate(similarity[idx]))

    # Sort by score descending, skip the first result (the movie itself)
    scores.sort(key=lambda x: x[1], reverse=True)
    top    = scores[1 : num_results + 1]

    return [(movies_df.iloc[i], score) for i, score in top]

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

/* ── Header ── */
.header {
    text-align: center;
    padding: 2rem 0 0.5rem;
}
.header h1 {
    font-family: 'Bebas Neue', cursive;
    font-size: 3.5rem;
    letter-spacing: 4px;
    color: #ffffff;
    margin: 0;
}
.header p {
    color: #888;
    font-size: 1rem;
    margin-top: 0.25rem;
}

/* ── Movie Card ── */
.card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 1rem;
    transition: transform 0.2s, border-color 0.2s;
}
.card:hover {
    transform: translateY(-4px);
    border-color: #e50914;
}

/* ── Movie Title inside card ── */
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 10px 0 6px;
    color: #ffffff;
}

/* ── Genre badges ── */
.badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
}
.badge {
    background: #e50914;
    color: #fff;
    font-size: 0.7rem;
    padding: 3px 9px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Score ── */
.score {
    color: #46c2ff;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 6px;
}

/* ── Cast line ── */
.cast {
    color: #aaa;
    font-size: 0.82rem;
    margin-bottom: 6px;
}

/* ── Overview ── */
.overview {
    color: #bbb;
    font-size: 0.82rem;
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── Select box + button ── */
div[data-testid="stSelectbox"] > label {
    font-size: 1rem;
    color: #ccc;
}
div[data-testid="stButton"] button {
    background: #e50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.85 !important;
}

/* ── Divider ── */
hr {
    border-color: #2a2a2a;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🎬 MOVIE RECOMMENDER</h1>
    <p>Pick a movie you love — we'll find what to watch next.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
#  MOVIE SELECTOR
# ─────────────────────────────────────────
movie_list     = sorted(movies_df["title"].dropna().unique().tolist())
selected_movie = st.selectbox("🎥 Choose a movie", movie_list)

col_btn, _ = st.columns([1, 3])
with col_btn:
    recommend_clicked = st.button("Find Similar Movies 🍿", use_container_width=True)

# ─────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────
if recommend_clicked:
    recommendations = get_recommendations(selected_movie)

    if not recommendations:
        st.warning("Sorry, couldn't find recommendations for that movie.")
    else:
        st.markdown(f"### Movies similar to **{selected_movie}**")
        st.markdown("---")

        # 3-column grid
        cols = st.columns(3)

        for i, (movie, score) in enumerate(recommendations):
            with cols[i % 3]:

                # Fetch poster image
                poster_url = fetch_poster(movie["title"])

                # Build genre badges HTML
                genre_badges = "".join(
                    f"<span class='badge'>{g}</span>"
                    for g in format_genres(movie["genres"])
                )

                # Build card HTML
                card_html = f"""
                <div class="card">
                    <img src="{poster_url}" style="width:100%; border-radius:8px;" />
                    <div class="card-title">{movie['title']}</div>
                    <div class="badges">{genre_badges}</div>
                    <div class="cast">🎭 {format_cast(movie['cast'])}</div>
                    <div class="overview">{format_overview(movie['overview'])}</div>
                    <div class="score">🎯 Similarity: {score:.3f}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "Powered by content-based filtering · Poster data from TMDB"
    "</p>",
    unsafe_allow_html=True,
)
