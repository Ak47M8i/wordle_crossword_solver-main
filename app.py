"""AI-Powered Wordle and Crossword Clue Solver Streamlit Application."""

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
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Wordle & Crossword Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Main Container & Modern Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header Hero Banner */
.hero-container {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.0rem;
    color: #94a3b8;
    margin-top: 6px;
    margin-bottom: 0px;
}

/* Wordle Tile Styling */
.wordle-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    justify-content: flex-start;
}

.wordle-tile {
    width: 48px;
    height: 48px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    border-radius: 6px;
    text-transform: uppercase;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    user-select: none;
}

.tile-green {
    background-color: #538d4e !important;
    border: 1px solid #457841;
}

.tile-yellow {
    background-color: #b59f3b !important;
    border: 1px solid #998632;
}

.tile-gray {
    background-color: #3a3a3c !important;
    border: 1px solid #2e2e30;
}

.tile-empty {
    background-color: #1e293b !important;
    border: 2px dashed #475569;
    color: #94a3b8 !important;
}

/* Crossword Letter Slots */
.crossword-slot-container {
    display: flex;
    gap: 6px;
    margin: 10px 0;
    flex-wrap: wrap;
}

.crossword-slot {
    width: 40px;
    height: 40px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    border-radius: 4px;
    background: #0f172a;
    border: 1px solid #38bdf8;
    color: #38bdf8;
}

.crossword-slot.blank {
    background: #1e293b;
    border: 1px dashed #64748b;
    color: #64748b;
}

/* Metric and Info Badges */
.badge-green {
    background-color: rgba(83, 141, 78, 0.2);
    color: #4ade80;
    border: 1px solid rgba(83, 141, 78, 0.4);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
}

.badge-yellow {
    background-color: rgba(181, 159, 59, 0.2);
    color: #facc15;
    border: 1px solid rgba(181, 159, 59, 0.4);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
}

.badge-blue {
    background-color: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Result Card */
.result-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.result-card:hover {
    border-color: #38bdf8;
    transform: translateY(-1px);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Crossword Clue Matcher Engine...")
def get_crossword_engine() -> CrosswordEngine:
    """Initializes and caches the Crossword TF-IDF search engine."""
    return CrosswordEngine()


@st.cache_data(show_spinner=False)
def load_wordle_dictionaries() -> Tuple[List[str], set]:
    """Loads and caches the target and allowed Wordle dictionaries."""
    targets = get_target_words()
    valid_set = get_all_valid_words()
    return targets, valid_set


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "wordle_history" not in st.session_state:
    st.session_state.wordle_history = []  # List of {"guess": str, "pattern": tuple[int, ...]}

if "sim_history" not in st.session_state:
    st.session_state.sim_history = None

if "cw_clue_input" not in st.session_state:
    st.session_state.cw_clue_input = "Egyptian queen for short"

if "cw_pattern_input" not in st.session_state:
    st.session_state.cw_pattern_input = "C _ _ O"

if "cw_length_input" not in st.session_state:
    st.session_state.cw_length_input = 4


# ---------------------------------------------------------
# Helper Functions for UI Rendering
# ---------------------------------------------------------
def render_tiles_html(word: str, pattern: Tuple[int, ...]) -> str:
    """Renders HTML for a row of Wordle feedback tiles."""
    html_parts = ['<div class="wordle-row">']
    class_map = {GRAY: "tile-gray", YELLOW: "tile-yellow", GREEN: "tile-green"}
    for ch, code in zip(word.upper(), pattern):
        cls_name = class_map.get(code, "tile-empty")
        html_parts.append(f'<div class="wordle-tile {cls_name}">{html.escape(ch)}</div>')
    html_parts.append('</div>')
    return "".join(html_parts)


def render_pattern_slots_html(pattern: str, length: int) -> str:
    """Renders visual character slots for a crossword query pattern."""
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
# Sidebar Navigation & General Metrics
# ---------------------------------------------------------
target_words, valid_words_set = load_wordle_dictionaries()
cw_engine = get_crossword_engine()

with st.sidebar:
    st.markdown("## 🧠 Mode Selection")
    mode = st.radio(
        label="Select Application Mode",
        options=["Wordle Solver & Assistant", "Crossword Clue Matcher"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Engine Status")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Wordle Secrets", f"{len(target_words):,}")
    with col_sb2:
        st.metric("Crossword Clues", f"{len(cw_engine.entries):,}")

    st.caption("Powered by Shannon Entropy & TF-IDF Semantic Vectorization")

    st.markdown("---")
    st.markdown("### 💡 Quick Guide")
    if mode == "Wordle Solver & Assistant":
        st.markdown(
            """
            - **Interactive Assistant**: Input your guess and tile colors (🟩 Green, 🟨 Yellow, ⬛ Gray) to compute the mathematically optimal next move.
            - **AI Simulation Mode**: Watch the AI automatically solve any hidden secret word step-by-step using Information Theory.
            """
        )
    else:
        st.markdown(
            """
            - **Pattern Syntax**: Use `_`, `.` or `?` for blank letters (e.g. `C _ _ T` or `..T`).
            - **Semantic Matching**: Enter natural language clues to search our dictionary & clue corpus via TF-IDF cosine similarity.
            """
        )


# =========================================================
# MODE 1: WORDLE SOLVER & ASSISTANT
# =========================================================
if mode == "Wordle Solver & Assistant":
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">🧩 AI Wordle Solver & Assistant</h1>
            <p class="hero-subtitle">
                Harness Information Theory and Shannon Entropy to calculate optimal guesses, eliminate possibilities, and simulate autonomous gameplay.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    wordle_tab_live, wordle_tab_sim, wordle_tab_theory = st.tabs([
        "🎯 Interactive Live Assistant",
        "🤖 Autonomous AI Simulation",
        "📐 Information Theory & Top Openers"
    ])

    # -----------------------------------------------------
    # TAB 1A: INTERACTIVE LIVE ASSISTANT
    # -----------------------------------------------------
    with wordle_tab_live:
        st.markdown("### 🎮 Live Game Board & Guess Input")
        st.info("Play along with your current Wordle game. Enter each guess and the feedback colors you received to receive real-time optimal recommendations.")

        col_board, col_recs = st.columns([1, 1.2], gap="large")

        # Compute remaining candidates based on history
        current_candidates = list(target_words)
        for step in st.session_state.wordle_history:
            current_candidates = filter_candidates(current_candidates, step["guess"], step["pattern"])

        with col_board:
            st.markdown("#### 📋 Current Game Board")
            if not st.session_state.wordle_history:
                st.markdown(
                    """
                    <div style="background: #1e293b; padding: 20px; border-radius: 8px; border: 1px dashed #475569; text-align: center; color: #94a3b8;">
                        No guesses entered yet. Input your first guess below or pick an optimal opener from the recommendations!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                for idx, step in enumerate(st.session_state.wordle_history, 1):
                    tiles_code = render_tiles_html(step["guess"], step["pattern"])
                    st.markdown(f"**Turn {idx}:** {step['guess']}", unsafe_allow_html=True)
                    st.markdown(tiles_code, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### ➕ Add New Guess")

            with st.form("add_guess_form", clear_on_submit=False):
                input_guess_word = st.text_input(
                    "5-Letter Guess Word",
                    value="",
                    max_chars=5,
                    placeholder="e.g. CRANE",
                    help="Enter a valid 5-letter English word."
                ).strip().upper()

                st.markdown("**Letter Feedback Colors:**")
                st.caption("Select the color feedback for each letter (0=⬛ Gray, 1=🟨 Yellow, 2=🟩 Green)")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                color_options = ["⬛ Gray", "🟨 Yellow", "🟩 Green"]
                
                with c1:
                    fb0 = st.selectbox("Pos 1", color_options, index=0, key="fb_0")
                with c2:
                    fb1 = st.selectbox("Pos 2", color_options, index=0, key="fb_1")
                with c3:
                    fb2 = st.selectbox("Pos 3", color_options, index=0, key="fb_2")
                with c4:
                    fb3 = st.selectbox("Pos 4", color_options, index=0, key="fb_3")
                with c5:
                    fb4 = st.selectbox("Pos 5", color_options, index=0, key="fb_4")

                # Quick text code alternative
                quick_code = st.text_input(
                    "Or Quick Code (Optional)",
                    placeholder="e.g. 01200 or GYBGG (G=Green, Y=Yellow, B=Gray)",
                    help="You can also type a 5-char code directly."
                ).strip().upper()

                submitted = st.form_submit_button("📥 Submit Guess Feedback", use_container_width=True)

                if submitted:
                    if len(input_guess_word) != 5 or not input_guess_word.isalpha():
                        st.error("Please enter a valid 5-letter word consisting of letters only.")
                    else:
                        # Determine feedback pattern
                        if quick_code and len(quick_code) == 5:
                            pattern_list = []
                            valid_code = True
                            for ch in quick_code:
                                if ch in ['G', '2']:
                                    pattern_list.append(GREEN)
                                elif ch in ['Y', '1']:
                                    pattern_list.append(YELLOW)
                                elif ch in ['B', 'X', '0', '_', 'G']:
                                    pattern_list.append(GRAY)
                                else:
                                    valid_code = False
                                    break
                            if valid_code and len(pattern_list) == 5:
                                pattern_tuple = tuple(pattern_list)
                            else:
                                st.warning("Quick code had unrecognized characters, used dropdown selections instead.")
                                color_to_val = {"⬛ Gray": GRAY, "🟨 Yellow": YELLOW, "🟩 Green": GREEN}
                                pattern_tuple = (color_to_val[fb0], color_to_val[fb1], color_to_val[fb2], color_to_val[fb3], color_to_val[fb4])
                        else:
                            color_to_val = {"⬛ Gray": GRAY, "🟨 Yellow": YELLOW, "🟩 Green": GREEN}
                            pattern_tuple = (color_to_val[fb0], color_to_val[fb1], color_to_val[fb2], color_to_val[fb3], color_to_val[fb4])

                        # Append to history
                        st.session_state.wordle_history.append({
                            "guess": input_guess_word,
                            "pattern": pattern_tuple
                        })
                        st.rerun()

            # Board action buttons
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("↩️ Undo Last Guess", use_container_width=True, disabled=len(st.session_state.wordle_history) == 0):
                    if st.session_state.wordle_history:
                        st.session_state.wordle_history.pop()
                        st.rerun()
            with btn_col2:
                if st.button("🔄 Reset Board", use_container_width=True, disabled=len(st.session_state.wordle_history) == 0):
                    st.session_state.wordle_history = []
                    st.rerun()

        with col_recs:
            st.markdown("#### 🧠 Real-Time Recommendation Engine")
            
            # Display candidate metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Remaining Candidates", f"{len(current_candidates):,}")
            with m_col2:
                if len(target_words) > 0:
                    pct_elim = ((len(target_words) - len(current_candidates)) / len(target_words)) * 100
                    st.metric("Elimination Rate", f"{pct_elim:.1f}%")

            if len(current_candidates) == 0:
                st.error("⚠️ No words in the dictionary match this exact combination of feedback patterns! Please verify your inputs or click 'Undo Last Guess'.")
            elif len(current_candidates) == 1:
                sole_word = current_candidates[0]
                st.success(f"🎉 **Guaranteed Solution Found:** The secret word is **{sole_word}**!")
                st.markdown(render_tiles_html(sole_word, (GREEN, GREEN, GREEN, GREEN, GREEN)), unsafe_allow_html=True)
            else:
                # Show top recommendations
                with st.spinner("Calculating Shannon Entropy rankings across possibilities..."):
                    recommendations = rank_next_guesses(
                        candidate_secrets=current_candidates,
                        allowed_vocab=list(target_words),
                        top_n=10,
                        prioritize_possible=True
                    )

                if recommendations:
                    st.markdown("##### 🏆 Optimal Next Guesses (by Information Gain)")
                    
                    rec_data = []
                    for rank, r in enumerate(recommendations, 1):
                        status = "🎯 Possible Secret" if r["is_possible"] else "💡 Strategic Burner"
                        rec_data.append({
                            "Rank": rank,
                            "Word": r["word"],
                            "Entropy (bits)": f"{r['entropy']:.3f}",
                            "Exp. Remaining": f"{r['expected_remaining']:.1f}",
                            "Win Chance": f"{r['win_probability']:.1f}%",
                            "Type": status
                        })

                    rec_df = pd.DataFrame(rec_data)
                    st.dataframe(rec_df, hide_index=True, use_container_width=True)

                    top_word = recommendations[0]["word"]
                    top_h = recommendations[0]["entropy"]
                    st.markdown(
                        f"""
                        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; border-radius: 8px; padding: 12px; margin-top: 10px;">
                            <strong>💡 AI Recommendation:</strong> Play <strong>{top_word}</strong> for an expected information gain of <strong>{top_h:.3f} bits</strong>.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Expandable view of all remaining candidates
                with st.expander(f"🔍 View all {len(current_candidates)} remaining candidate words"):
                    pills_html = " ".join([f'<span class="badge-blue" style="margin: 2px; display: inline-block;">{w}</span>' for w in sorted(current_candidates)[:200]])
                    if len(current_candidates) > 200:
                        pills_html += f" <em>...and {len(current_candidates) - 200} more</em>"
                    st.markdown(pills_html, unsafe_allow_html=True)

    # -----------------------------------------------------
    # TAB 1B: AUTONOMOUS AI SIMULATION MODE
    # -----------------------------------------------------
    with wordle_tab_sim:
        st.markdown("### 🤖 Autonomous AI Wordle Simulation")
        st.write("Pick or generate a hidden secret word. The AI will autonomously deduce and solve the puzzle step-by-step using Shannon entropy.")

        sim_col1, sim_col2 = st.columns([1, 1], gap="medium")
        with sim_col1:
            secret_selection_mode = st.radio(
                "Secret Word Source",
                ["Choose from Popular Targets", "Random Secret Word", "Enter Custom Secret Word"],
                horizontal=True
            )

            if secret_selection_mode == "Choose from Popular Targets":
                popular_samples = ["SLATE", "CRANE", "ROBOT", "VIVID", "JAZZY", "KNIFE", "LIGHT", "ZEBRA", "COBRA", "APPLE"]
                chosen_secret = st.selectbox("Select Secret Word", popular_samples, index=0)
            elif secret_selection_mode == "Random Secret Word":
                if st.button("🎲 Draw Random Word"):
                    st.session_state.random_secret = random.choice(target_words)
                if "random_secret" not in st.session_state:
                    st.session_state.random_secret = random.choice(target_words)
                chosen_secret = st.session_state.random_secret
                st.info(f"Target selected: **{chosen_secret}**")
            else:
                custom_word_input = st.text_input("Enter Any 5-Letter Secret Word", value="GHOST", max_chars=5).strip().upper()
                chosen_secret = custom_word_input

        with sim_col2:
            opener_choice = st.selectbox(
                "AI Starting Opener",
                ["SOARE (Optimal: 5.885 bits)", "ROATE (5.885 bits)", "RAISE (5.878 bits)", "SLATE (5.856 bits)", "CRANE (5.740 bits)", "AUDIO (Vowel-Heavy: 5.032 bits)"],
                index=0
            )
            opener_word = opener_choice.split()[0]
            max_turns_allowed = st.slider("Max Allowed Turns", min_value=4, max_value=8, value=6)

            st.write("")
            run_sim_btn = st.button("▶️ Launch AI Simulation", type="primary", use_container_width=True)

        if run_sim_btn:
            if len(chosen_secret) != 5 or not chosen_secret.isalpha():
                st.error("The secret word must be a valid 5-letter alphabetic word.")
            else:
                with st.spinner("AI is evaluating feedback and calculating entropy distributions..."):
                    sim_result = simulate_game(
                        secret_word=chosen_secret,
                        first_guess=opener_word,
                        max_turns=max_turns_allowed
                    )
                    st.session_state.sim_history = sim_result

        # Display simulation results if available
        if st.session_state.sim_history:
            sim_res = st.session_state.sim_history
            st.markdown("---")
            st.markdown("#### 🎬 Step-by-Step Simulation Replay")

            won = sim_res["won"]
            turns_taken = sim_res["turns_taken"]
            secret = sim_res["secret_word"]

            if won:
                st.success(f"🏆 **Victory!** The AI solved the puzzle in **{turns_taken}** turns for the target word **{secret}**!")
            else:
                st.error(f"❌ **Exceeded Limit:** The AI was unable to solve **{secret}** within {max_turns_allowed} turns.")

            # Step-by-step turns
            for step in sim_res["history"]:
                turn_num = step["turn"]
                guess_w = step["guess"]
                fb = step["feedback"]
                h_val = step["entropy"]
                rem_bef = step["remaining_before"]
                rem_aft = step["remaining_after"]
                is_win = step["is_correct"]

                with st.container():
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Turn {turn_num}: {guess_w}</span>
                                <div>
                                    <span class="badge-blue">Entropy: {h_val:.2f} bits</span>
                                    <span class="badge-green">Remaining: {rem_bef:,} &rarr; {rem_aft:,}</span>
                                </div>
                            </div>
                            {render_tiles_html(guess_w, fb)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Benchmark demonstration
            st.markdown("---")
            st.markdown("##### 📈 Quick AI Benchmark Test")
            st.caption("Test the solver's consistency across 5 random secret words simultaneously.")
            if st.button("🚀 Run 5-Word Random Benchmark"):
                bench_samples = random.sample(target_words, 5)
                bench_records = []
                total_turns = 0
                for w in bench_samples:
                    res = simulate_game(secret_word=w, first_guess="SOARE", max_turns=6)
                    bench_records.append({
                        "Secret Word": w,
                        "Status": "Solved ✅" if res["won"] else "Failed ❌",
                        "Turns Taken": res["turns_taken"],
                        "Path": " -> ".join([s["guess"] for s in res["history"]])
                    })
                    total_turns += res["turns_taken"]

                avg_turns = total_turns / len(bench_samples)
                st.dataframe(pd.DataFrame(bench_records), hide_index=True, use_container_width=True)
                st.metric("Benchmark Average Turns", f"{avg_turns:.2f} turns")

    # -----------------------------------------------------
    # TAB 1C: INFORMATION THEORY EXPLANATION
    # -----------------------------------------------------
    with wordle_tab_theory:
        st.markdown("### 📐 How Shannon Entropy Powers the Solver")
        st.markdown(
            r"""
            Claude Shannon's Information Theory provides the mathematical foundation for Wordle optimization.

            #### 1. The Math of Shannon Entropy
            Each Wordle feedback consists of 5 colored tiles with 3 possibilities each (Green, Yellow, Gray), producing $3^5 = 243$ possible patterns $p$.

            For any guess $g$ evaluated against the remaining secret word candidates $C$:
            - The candidate set is partitioned into disjoint buckets $C_p$ corresponding to each pattern $p$.
            - The probability of receiving pattern $p$ is $P(p) = \frac{|C_p|}{|C|}$.
            - The **expected information gain (entropy)** $H(g)$ measured in bits is:

            $$H(g) = -\sum_{p} P(p) \log_2(P(p)) = \sum_{p} \frac{|C_p|}{|C|} \log_2\left(\frac{|C|}{|C_p|}\right)$$

            A higher entropy means the guess spreads the candidate words evenly across diverse feedback patterns, guaranteeing the smallest possible expected candidate pool on the following turn!
            """
        )

        st.markdown("#### 🏆 Globally Precomputed Top Wordle Openers")
        openers_df = pd.DataFrame(OPTIMAL_OPENERS)
        openers_df.columns = ["Word", "Shannon Entropy (bits)", "Expected Remaining Words", "Can Be Answer?"]
        st.dataframe(openers_df, hide_index=True, use_container_width=True)


# =========================================================
# MODE 2: CROSSWORD CLUE MATCHER
# =========================================================
else:
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">🔍 Crossword Clue Matcher & Regex Engine</h1>
            <p class="hero-subtitle">
                Solve cryptic and standard crossword clues using constraint pattern matching, TF-IDF semantic vector similarity, and word frequency heuristics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧩 Clue & Pattern Query")

    # Quick presets
    st.markdown("**⚡ Quick Example Clues:**")
    preset_cols = st.columns(6)
    presets = [
        ("Egyptian Queen", "Egyptian queen for short", "C _ _ O", 4),
        ("Capital of France", "Capital of France and City of Light", "P _ _ _ S", 5),
        ("Feline Companion", "Feline companion or pet", "C _ _", 3),
        ("Opposite of Day", "Opposite of day when sun is down", "N _ _ _ T", 5),
        ("Large Water Body", "Large expansive body of salt water", "O _ _ _ N", 5),
        ("Red Gemstone", "Precious deep red gemstone", "R _ _ Y", 4),
    ]

    for pcol, (lbl, p_clue, p_pat, p_len) in zip(preset_cols, presets):
        with pcol:
            if st.button(lbl, use_container_width=True):
                st.session_state.cw_clue_input = p_clue
                st.session_state.cw_pattern_input = p_pat
                st.session_state.cw_length_input = p_len
                st.rerun()

    st.markdown("")

    with st.form("crossword_search_form"):
        col_c1, col_c2, col_c3 = st.columns([1.8, 1.2, 0.8], gap="medium")

        with col_c1:
            clue_query = st.text_input(
                "Crossword Clue",
                value=st.session_state.cw_clue_input,
                placeholder="e.g. Capital of France, Feline pet, Mona ___",
                help="Type any crossword clue or concept."
            ).strip()

        with col_c2:
            pattern_query = st.text_input(
                "Pattern (Blanks: _ , . , ?)",
                value=st.session_state.cw_pattern_input,
                placeholder="e.g. C _ _ T or P..IS",
                help="Specify known letters and blanks."
            ).strip()

        with col_c3:
            length_query = st.number_input(
                "Word Length",
                min_value=0,
                max_value=15,
                value=st.session_state.cw_length_input,
                help="0 = Auto-detect from pattern length"
            )

        submit_search = st.form_submit_button("🔍 Find Matching Answers", type="primary", use_container_width=True)

    # Live pattern preview
    if pattern_query:
        st.markdown("**Visual Letter Pattern Slots:**")
        slots_html = render_pattern_slots_html(pattern_query, length_query)
        st.markdown(slots_html, unsafe_allow_html=True)

    if submit_search or pattern_query or clue_query:
        with st.spinner("Searching corpus, compiling regex, and calculating TF-IDF cosine similarity..."):
            results = cw_engine.search_candidates(
                clue=clue_query,
                pattern=pattern_query,
                length=length_query if length_query > 0 else None,
                top_k=25
            )

        st.markdown("---")
        st.markdown(f"### 📋 Candidate Answers ({len(results)} matches found)")

        if not results:
            st.warning("No candidate answers in the dictionary matched both your pattern and length constraints. Try relaxing the pattern or checking the word length.")
        else:
            col_list, col_details = st.columns([1.3, 0.9], gap="large")

            with col_list:
                for idx, item in enumerate(results, 1):
                    ans = item["answer"]
                    score = item["score"]
                    sim = item["semantic_similarity"]
                    category = item["category"]
                    best_clue = item["best_clue"]
                    source = item["source"]

                    # Visual score bar color
                    if score >= 75:
                        badge_cls = "badge-green"
                    elif score >= 50:
                        badge_cls = "badge-yellow"
                    else:
                        badge_cls = "badge-blue"

                    sim_text = f" • Semantic Match: {sim:.1f}%" if sim is not None else ""

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <span style="font-size: 1.25rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #38bdf8;">{ans}</span>
                                    <span class="{badge_cls}">Score: {score:.1f}%</span>
                                    <span class="badge-blue">{len(ans)} Letters</span>
                                    <span style="font-size: 0.8rem; color: #94a3b8;">({category})</span>
                                </div>
                                <span style="font-size: 0.8rem; color: #64748b;">Source: {source}</span>
                            </div>
                            <div style="margin-top: 8px; color: #cbd5e1; font-size: 0.92rem;">
                                <strong>Corpus Clue / Definition:</strong> <em>"{best_clue}"</em>{sim_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col_details:
                st.markdown("#### 📊 Candidate Summary Table")
                df_data = []
                for idx, item in enumerate(results, 1):
                    df_data.append({
                        "Rank": idx,
                        "Answer": item["answer"],
                        "Confidence": f"{item['score']:.1f}%",
                        "Category": item["category"]
                    })
                st.dataframe(pd.DataFrame(df_data), hide_index=True, use_container_width=True)

                st.markdown(
                    """
                    <div style="background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #334155;">
                        <h5 style="margin-top: 0; color: #f8fafc;">🔍 How Crossword Matching Works</h5>
                        <ul style="padding-left: 20px; color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                            <li><strong>Pattern Parsing:</strong> Converted to an anchored regular expression (e.g. <code>^C..T$</code>).</li>
                            <li><strong>TF-IDF Vectorization:</strong> Clues are transformed into n-gram TF-IDF vectors, and cosine similarity is computed against hundreds of curated crossword definitions.</li>
                            <li><strong>Hybrid Scoring:</strong> Blends semantic match confidence, word frequency, and direct token presence.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Cheatsheet expander
    with st.expander("📖 Crossword Regex & Pattern Cheatsheet"):
        st.markdown(
            """
            | Pattern | Meaning | Example Matches |
            | :--- | :--- | :--- |
            | `C _ _ T` | 4-letter word starting with C and ending with T | `CART`, `COAT`, `COST` |
            | `P . . . S` | 5-letter word starting with P and ending with S | `PARIS`, `PLUMS`, `PIRES` |
            | `^A[A-Z]{3}E$` | Advanced regex: starts with A, 3 letters, ends with E | `APPLE`, `AGREE`, `ALONE` |
            | `..T` | 3-letter word ending with T | `CAT`, `BAT`, `HAT` |
            """
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
        AI-Powered Wordle and Crossword Clue Solver • Shannon Entropy & TF-IDF Semantic Search Engine
    </div>
    """,
    unsafe_allow_html=True
)
