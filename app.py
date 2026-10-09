"""Gamified AI-Powered Wordle and Crossword Clue Solver Streamlit Application."""

import html
import random
import re
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

from crossword_engine import CrosswordEngine
from data.wordle_words import (
    OPTIMAL_OPENERS,
    get_all_valid_words,
    get_target_words,
)
from wordle_engine import (
    COLOR_ICONS,
    COLOR_NAMES,
    GRAY,
    GREEN,
    HEX_COLORS,
    YELLOW,
    calculate_entropy,
    calculate_frequency_score,
    compute_feedback,
    filter_candidates,
    rank_next_guesses,
    simulate_game,
)

# ---------------------------------------------------------
# Page Configuration & Gamified Arcade CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Arcade AI: Wordle & Crossword Master",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

GAMIFIED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800;900&family=JetBrains+Mono:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Arcade Hero Header */
.arcade-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    border: 2px solid #6366f1;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.25);
    position: relative;
    overflow: hidden;
}
.arcade-title {
    font-size: 2.4rem;
    font-weight: 900;
    margin: 0;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.arcade-subtitle {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-top: 8px;
    margin-bottom: 0px;
    font-weight: 400;
}

/* Gamified Stat Badges */
.stat-pill {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
}
.stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

/* Wordle Tile Styles */
.wordle-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    justify-content: flex-start;
}
.wordle-tile {
    width: 52px;
    height: 52px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    border-radius: 8px;
    text-transform: uppercase;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    user-select: none;
    transition: transform 0.2s ease;
}
.wordle-tile:hover {
    transform: scale(1.05);
}
.tile-green { background-color: #22c55e !important; border: 2px solid #16a34a; }
.tile-yellow { background-color: #eab308 !important; border: 2px solid #ca8a04; color: #1e293b !important; }
.tile-gray { background-color: #334155 !important; border: 2px solid #1e293b; color: #94a3b8 !important; }
.tile-empty { background-color: #0f172a !important; border: 2px dashed #475569; color: #64748b !important; }

/* Interactive Virtual Keyboard */
.kb-row {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-bottom: 6px;
}
.kb-key {
    background-color: #334155;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    text-align: center;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
.kb-green { background-color: #22c55e !important; color: #ffffff !important; }
.kb-yellow { background-color: #eab308 !important; color: #1e293b !important; }
.kb-gray { background-color: #1e293b !important; color: #475569 !important; border: 1px solid #1e1b4b; }

/* Crossword Slots */
.crossword-slot-container {
    display: flex;
    gap: 6px;
    margin: 12px 0;
    flex-wrap: wrap;
}
.crossword-slot {
    width: 44px;
    height: 44px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    border-radius: 8px;
    background: #0f172a;
    border: 2px solid #38bdf8;
    color: #38bdf8;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
}
.crossword-slot.blank {
    background: #1e293b;
    border: 2px dashed #475569;
    color: #64748b;
    box-shadow: none;
}

/* Arcade Result Cards */
.arcade-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
}
.arcade-card:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
}
</style>
"""
st.markdown(GAMIFIED_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Loading Crossword Matrix Engine...")
def get_crossword_engine() -> CrosswordEngine:
    return CrosswordEngine()

@st.cache_data(show_spinner=False)
def load_wordle_dictionaries() -> Tuple[List[str], set]:
    return get_target_words(), get_all_valid_words()

# ---------------------------------------------------------
# Session State Initialization (Gamification Stats)
# ---------------------------------------------------------
if "wordle_history" not in st.session_state:
    st.session_state.wordle_history = []
if "sim_history" not in st.session_state:
    st.session_state.sim_history = None
if "games_played" not in st.session_state:
    st.session_state.games_played = 0
if "win_streak" not in st.session_state:
    st.session_state.win_streak = 0
if "cw_clue_input" not in st.session_state:
    st.session_state.cw_clue_input = "Capital of France and City of Light"
if "cw_pattern_input" not in st.session_state:
    st.session_state.cw_pattern_input = "P _ _ _ S"
if "cw_length_input" not in st.session_state:
    st.session_state.cw_length_input = 5

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def render_tiles_html(word: str, pattern: Tuple[int, ...]) -> str:
    html_parts = ['<div class="wordle-row">']
    class_map = {GRAY: "tile-gray", YELLOW: "tile-yellow", GREEN: "tile-green"}
    for ch, code in zip(word.upper(), pattern):
        cls_name = class_map.get(code, "tile-empty")
        html_parts.append(f'<div class="wordle-tile {cls_name}">{html.escape(ch)}</div>')
    html_parts.append('</div>')
    return "".join(html_parts)

def render_virtual_keyboard(history: List[Dict[str, Any]]) -> str:
    """Computes letter states and renders a gamified virtual QWERTY keyboard."""
    letter_states = {} # 0: unplayed, 1: gray, 2: yellow, 3: green
    for step in history:
        g = step["guess"].upper()
        pat = step["pattern"]
        for ch, code in zip(g, pat):
            val = code + 1 # Convert 0,1,2 to 1,2,3
            current = letter_states.get(ch, 0)
            if val > current:
                if current == 3 and val != 3:
                    continue # Green overrides others
                letter_states[ch] = val

    rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
    kb_html = '<div style="margin: 15px 0;">'
    for r in rows:
        kb_html += '<div class="kb-row">'
        for ch in r:
            state = letter_states.get(ch, 0)
            cls = "kb-key"
            if state == 3:
                cls += " kb-green"
            elif state == 2:
                cls += " kb-yellow"
            elif state == 1:
                cls += " kb-gray"
            kb_html += f'<div class="{cls}">{ch}</div>'
        kb_html += '</div>'
    kb_html += '</div>'
    return kb_html

def render_pattern_slots_html(pattern: str, length: int) -> str:
    engine = CrosswordEngine
    compiled_regex, detected_len = engine.parse_pattern_to_regex(pattern, length)
    display_len = length if length > 0 else (detected_len if detected_len > 0 else len(pattern))
    clean_chars = []
    for ch in pattern:
        if ch.isalpha():
            clean_chars.append(ch.upper())
        elif ch in ['_', '.', '?', '*']:
            clean_chars.append('_')
    while len(clean_chars) < display_len:
        clean_chars.append('_')
    clean_chars = clean_chars[:display_len]
    html_parts = ['<div class="crossword-slot-container">']
    for ch in clean_chars:
        if ch == '_':
            html_parts.append('<div class="crossword-slot blank">_</div>')
        else:
            html_parts.append(f'<div class="crossword-slot">{html.escape(ch)}</div>')
    html_parts.append('</div>')
    return "".join(html_parts)

# ---------------------------------------------------------
# Sidebar Dashboard & Controls
# ---------------------------------------------------------
target_words, valid_words_set = load_wordle_dictionaries()
cw_engine = get_crossword_engine()

with st.sidebar:
    st.markdown("### 🎮 ARCADE SELECTOR")
    mode = st.radio(
        label="App Mode",
        options=["Wordle AI Battle & Solver", "Crossword Crypto Matcher"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 🏆 PLAYER STATS")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown(f"""<div class="stat-pill"><div class="stat-value">{st.session_state.games_played}</div><div class="stat-label">Played</div></div>""", unsafe_allow_html=True)
    with s_col2:
        st.markdown(f"""<div class="stat-pill"><div class="stat-value">{st.session_state.win_streak}</div><div class="stat-label">Streak 🔥</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚡ ENGINE SPECS")
    st.caption(f"• Wordle Secret Base: **{len(target_words):,} words**")
    st.caption(f"• Crossword Corpus: **{len(cw_engine.entries):,} entries**")
    st.caption("• Algorithms: Shannon Entropy & TF-IDF Cosine Similarity")

# =========================================================
# MODE 1: WORDLE AI BATTLE & SOLVER
# =========================================================
if mode == "Wordle AI Battle & Solver":
    st.markdown(
        """
        <div class="arcade-hero">
            <h1 class="arcade-title">🎯 Wordle Entropy Arena</h1>
            <p class="arcade-subtitle">
                Outsmart the secret word using Information Theory, or launch autonomous AI simulations to watch Shannon entropy decimate possibilities in real-time!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_live, tab_sim, tab_theory = st.tabs([
        "🎮 Interactive Live Game",
        "🤖 Autonomous AI Battle",
        "🧠 Entropy Mechanics"
    ])

    # --- TAB 1: LIVE INTERACTIVE GAME ---
    with tab_live:
        st.markdown("### 🕹️ Live Board & AI Assistant")
        st.write("Enter your word guesses and color feedback to get mathematically calculated optimal next moves.")

        col_board, col_recs = st.columns([1, 1.2], gap="large")
        
        current_candidates = list(target_words)
        for step in st.session_state.wordle_history:
            current_candidates = filter_candidates(current_candidates, step["guess"], step["pattern"])

        with col_board:
            st.markdown("#### 📋 Active Game Board")
            if not st.session_state.wordle_history:
                st.info("No guesses logged yet. Start by entering a guess below or pick 'SOARE' / 'CRANE'!")
            else:
                for idx, step in enumerate(st.session_state.wordle_history, 1):
                    st.markdown(f"**Turn {idx}:** `{step['guess']}`")
                    st.markdown(render_tiles_html(step["guess"], step["pattern"]), unsafe_allow_html=True)
                
                # Render Virtual Keyboard
                st.markdown("#### ⌨️ Virtual Letter Status")
                st.markdown(render_virtual_keyboard(st.session_state.wordle_history), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### ➕ Log Your Guess Feedback")
            with st.form("live_guess_form", clear_on_submit=False):
                guess_input = st.text_input("5-Letter Guess", max_chars=5, placeholder="e.g. CRANE").strip().upper()
                st.markdown("**Tile Colors:** (0 = ⬛ Gray, 1 = 🟨 Yellow, 2 = 🟩 Green)")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                opts = ["⬛ Gray", "🟨 Yellow", "🟩 Green"]
                with c1: f0 = st.selectbox("1", opts, index=0, key="k1")
                with c2: f1 = st.selectbox("2", opts, index=0, key="k2")
                with c3: f2 = st.selectbox("3", opts, index=0, key="k3")
                with c4: f3 = st.selectbox("4", opts, index=0, key="k4")
                with c5: f4 = st.selectbox("5", opts, index=0, key="k5")

                submitted = st.form_submit_button("🚀 Submit Guess & Compute Next Step", use_container_width=True)
                if submitted:
                    if len(guess_input) != 5 or not guess_input.isalpha():
                        st.error("Please enter a valid 5-letter alphabetic word.")
                    else:
                        c_map = {"⬛ Gray": GRAY, "🟨 Yellow": YELLOW, "🟩 Green": GREEN}
                        pat_tuple = (c_map[f0], c_map[f1], c_map[f2], c_map[f3], c_map[f4])
                        st.session_state.wordle_history.append({"guess": guess_input, "pattern": pat_tuple})
                        
                        # Check win condition
                        if pat_tuple == (GREEN, GREEN, GREEN, GREEN, GREEN):
                            st.session_state.games_played += 1
                            st.session_state.win_streak += 1
                            st.balloons()
                        st.rerun()

            b1, b2 = st.columns(2)
            with b1:
                if st.button("↩️ Undo Last Guess", use_container_width=True, disabled=len(st.session_state.wordle_history) == 0):
                    if st.session_state.wordle_history:
                        st.session_state.wordle_history.pop()
                        st.rerun()
            with b2:
                if st.button("🔄 Reset Board", use_container_width=True, disabled=len(st.session_state.wordle_history) == 0):
                    st.session_state.wordle_history = []
                    st.rerun()

        with col_recs:
            st.markdown("### 💡 AI Recommendation Engine")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Remaining Pool", f"{len(current_candidates):,}")
            with m2:
                if len(target_words) > 0:
                    elim = ((len(target_words) - len(current_candidates)) / len(target_words)) * 100
                    st.metric("Elimination Rate", f"{elim:.1f}%")

            if len(current_candidates) == 0:
                st.error("⚠️ No words match this exact combination. Check your inputs or undo the last guess.")
            elif len(current_candidates) == 1:
                sol = current_candidates[0]
                st.success(f"🎉 **Target Isolated:** The secret word is **{sol}**!")
                st.markdown(render_tiles_html(sol, (GREEN, GREEN, GREEN, GREEN, GREEN)), unsafe_allow_html=True)
            else:
                with st.spinner("Computing Shannon Entropy across candidate partitions..."):
                    recs = rank_next_guesses(candidate_secrets=current_candidates, allowed_vocab=list(target_words), top_n=8)
                
                if recs:
                    st.markdown("##### 🌟 Top Optimal Next Guesses")
                    rec_rows = []
                    for rank, r in enumerate(recs, 1):
                        st_type = "✨ Secret Match" if r["is_possible"] else "🔥 Strategic Burner"
                        rec_rows.append({
                            "Rank": rank,
                            "Word": r["word"],
                            "Entropy (bits)": f"{r['entropy']:.3f}",
                            "Win %": f"{r['win_probability']:.1f}%",
                            "Type": st_type
                        })
                    st.dataframe(pd.DataFrame(rec_rows), hide_index=True, use_container_width=True)
                    
                    top_w = recs[0]["word"]
                    top_h = recs[0]["entropy"]
                    st.markdown(
                        f"""
                        <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid #6366f1; border-radius: 10px; padding: 14px; margin-top: 12px;">
                            <strong>⚡ Best AI Move:</strong> Play <strong>{top_w}</strong> to harvest <strong>{top_h:.3f} bits</strong> of expected information!
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # --- TAB 2: AUTONOMOUS SIMULATION ---
    with tab_sim:
        st.markdown("### 🤖 Autonomous AI Battle Simulation")
        st.write("Watch the AI algorithm play against a hidden word autonomously using Shannon Entropy.")

        s_col1, s_col2 = st.columns([1, 1], gap="medium")
        with s_col1:
            sec_mode = st.radio("Secret Source", ["Popular Targets", "Random Secret", "Custom Secret"], horizontal=True)
            if sec_mode == "Popular Targets":
                secret_word = st.selectbox("Select Word", ["SLATE", "CRANE", "ROBOT", "VIVID", "JAZZY", "GHOST", "ZEBRA"])
            elif sec_mode == "Random Secret":
                if st.button("🎲 Roll Random Secret"):
                    st.session_state.rand_sec = random.choice(target_words)
                secret_word = st.session_state.get("rand_sec", "CRANE")
                st.info(f"Secret Target Loaded: **{secret_word}**")
            else:
                secret_word = st.text_input("Custom Secret Word", "MAGIC", max_chars=5).strip().upper()

        with s_col2:
            opener = st.selectbox("AI Opening Move", ["SOARE (5.885 bits)", "ROATE (5.885 bits)", "RAISE (5.878 bits)", "SLATE (5.856 bits)"]).split()[0]
            max_t = st.slider("Max Turns Limit", 4, 8, 6)
            run_sim = st.button("🚀 Launch Autonomous Simulation", type="primary", use_container_width=True)

        if run_sim:
            if len(secret_word) != 5 or not secret_word.isalpha():
                st.error("Secret must be a valid 5-letter word.")
            else:
                with st.spinner("AI calculating partition probabilities..."):
                    res = simulate_game(secret_word=secret_word, first_guess=opener, max_turns=max_t)
                    st.session_state.sim_history = res

        if st.session_state.sim_history:
            sim = st.session_state.sim_history
            st.markdown("---")
            if sim["won"]:
                st.success(f"🏆 **AI Victory!** Solved **{sim['secret_word']}** in **{sim['turns_taken']} turns**!")
                st.balloons()
            else:
                st.error(f"❌ AI failed to solve **{sim['secret_word']}** within {max_t} turns.")

            for stp in sim["history"]:
                st.markdown(
                    f"""
                    <div class="arcade-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <strong>Turn {stp['turn']}: <code>{stp['guess']}</code></strong>
                            <div>
                                <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem;">Entropy: {stp['entropy']:.2f} bits</span>
                                <span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem;">Remaining: {stp['remaining_after']:,}</span>
                            </div>
                        </div>
                        {render_tiles_html(stp['guess'], stp['feedback'])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # --- TAB 3: THEORY ---
    with tab_theory:
        st.markdown("### 🧠 The Math of Shannon Entropy")
        st.markdown(
            r"""
            Claude Shannon's Information Theory dictates that the optimal guess is the one that maximizes expected information gain across candidate partitions:
            $$H(g) = -\sum_{p} P(p) \log_2(P(p)) = \sum_{p} \frac{\vert{}C_p\vert{}}{\vert{}C\vert{}} \log_2\left(\frac{\vert{}C\vert{}}{\vert{}C_p\vert{}}\right)$$
            """
        )
        st.markdown("#### 🌟 Globally Precomputed Optimal Openers")
        st.dataframe(pd.DataFrame(OPTIMAL_OPENERS), hide_index=True, use_container_width=True)

# =========================================================
# MODE 2: CROSSWORD CRYPTO MATCHER
# =========================================================
 else:
    st.markdown(
        """
        <div class="arcade-hero">
            <h1 class="arcade-title">🧩 Crossword Crypto Matcher</h1>
            <p class="arcade-subtitle">
                Decode cryptic and standard crossword clues using anchored regex pattern matching and TF-IDF semantic vector similarity.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔥 Quick Preset Clues")
    p_cols = st.columns(6)
    presets = [
        ("Paris", "Capital of France and City of Light", "P _ _ _ S", 5),
        ("Cleo", "Egyptian queen for short", "C . . O", 4),
        ("Cat", "Feline companion or pet", "C _ _", 3),
        ("Night", "Opposite of day when sun is down", "N _ _ _ T", 5),
        ("Ocean", "Large expansive body of salt water", "O _ _ _ N", 5),
        ("Ruby", "Precious deep red gemstone", "R _ _ Y", 4),
    ]
    for pcol, (lbl, p_clue, p_pat, p_len) in zip(p_cols, presets):
        with pcol:
            if st.button(lbl, use_container_width=True):
                st.session_state.cw_clue_input = p_clue
                st.session_state.cw_pattern_input = p_pat
                st.session_state.cw_length_input = p_len
                st.rerun()

    st.markdown("")
    with st.form("cw_form"):
        cc1, cc2, cc3 = st.columns([1.8, 1.2, 0.8], gap="medium")
        with cc1:
            clue_q = st.text_input("Crossword Clue / Concept", value=st.session_state.cw_clue_input).strip()
        with cc2:
            pat_q = st.text_input("Pattern (Blanks: _ or .)", value=st.session_state.cw_pattern_input).strip()
        with cc3:
            len_q = st.number_input("Length", min_value=0, max_value=15, value=st.session_state.cw_length_input)
        
        search_btn = st.form_submit_button("🔍 Decode Crossword Match", type="primary", use_container_width=True)

    if pat_q:
        st.markdown("**Visual Letter Pattern Slots:**")
        st.markdown(render_pattern_slots_html(pat_q, len_q), unsafe_allow_html=True)

    if search_btn or pat_q or clue_q:
        with st.spinner("Analyzing semantic vectors and compiling regex filters..."):
            results = cw_engine.search_candidates(clue=clue_q, pattern=pat_q, length=len_q if len_q > 0 else None, top_k=20)

        st.markdown("---")
        st.markdown(f"### 🎯 Matching Answers ({len(results)} found)")
        if not results:
            st.warning("No matches found for this pattern and clue combination. Try relaxing constraints.")
        else:
            r_col1, r_col2 = st.columns([1.3, 0.9], gap="large")
            with r_col1:
                for item in results:
                    ans = item["answer"]
                    score = item["score"]
                    sim = item["semantic_similarity"]
                    cat = item["category"]
                    def_text = item["best_clue"]
                    
                    st.markdown(
                        f"""
                        <div class="arcade-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <span style="font-size: 1.3rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #38bdf8;">{ans}</span>
                                    <span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.82rem;">Score: {score:.1f}%</span>
                                    <span style="font-size: 0.8rem; color: #94a3b8;">({cat})</span>
                                </div>
                                <span style="font-size: 0.8rem; color: #64748b;">{item['source']}</span>
                            </div>
                            <div style="margin-top: 8px; color: #cbd5e1; font-size: 0.9rem;">
                                <strong>Corpus Definition:</strong> <em>"{def_text}"</em> {f" | Semantic Match: {sim:.1f}%" if sim else ""}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            with r_col2:
                st.markdown("#### 📊 Candidate Summary")
                summary_data = [{"Answer": i["answer"], "Confidence": f"{i['score']:.1f}%", "Category": i["category"]} for i in results]
                st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px 0;">
        🎮 Arcade AI Solver | Powered by Shannon Entropy & TF-IDF Semantic Vectorization
    </div>
    """,
    unsafe_allow_html=True
)