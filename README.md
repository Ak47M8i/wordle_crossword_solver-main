# 🧠 AI-Powered Wordle and Crossword Clue Solver

An advanced, interactive web application built with Python and Streamlit that solves Wordle puzzles using Information Theory (Shannon Entropy) and decrypts crossword clues using TF-IDF semantic vectorization and regex pattern matching.

---

## 🌟 Key Features

### 1. 🧩 Wordle Solver & Assistant
- **Interactive Live Assistant:**
  - Enter your past guesses and tile colors (🟩 Green, 🟨 Yellow, ⬛ Gray).
  - Supports both interactive color dropdowns and rapid code input (`GYBGG` or `21022`).
  - Real-time calculation of remaining valid secret words.
  - **Information Theory Optimizer:** Computes Shannon entropy ($H(g) = -\sum P(p) \log_2 P(p)$) in bits for all candidate words to recommend the mathematically optimal next guess.
  - Distinguishes between **Target Candidates** (immediate win opportunity) and **Strategic Burners** (maximum partition entropy).
- **Autonomous AI Simulation Mode:**
  - Let the AI play against any hidden secret word (custom, popular presets, or randomly drawn).
  - Full step-by-step game trace displaying turns taken, colored tiles, entropy values, and remaining candidate counts.
  - **Batch Benchmark:** Run automated benchmarks across multiple random words to verify solver efficiency (typically solving in 3.4–3.6 turns).
- **Theory & Top Openers:**
  - Complete mathematical explanation of Shannon entropy.
  - Precomputed reference table of top global openers (`SOARE`, `ROATE`, `RAISE`, `SLATE`, `CRANE`).

### 2. 🔍 Crossword Clue Matcher
- **Flexible Pattern Syntax:**
  - Supports underscores (`C _ _ T`), dots (`C..T`), question marks (`C??T`), or full regular expressions.
  - Dynamic character slot preview highlighting filled and blank positions.
- **TF-IDF Semantic Vectorization:**
  - Vectorizes natural language clues using `scikit-learn` n-gram TF-IDF and computes cosine similarity against a curated corpus of clues, synonyms, and dictionary definitions.
- **Hybrid Confidence Scoring:**
  - Combines semantic similarity (60%), word commonality/frequency (25%), and exact keyword presence (15%) into an intuitive 0–100% confidence score.
- **Rich Result Display:**
  - Category tags (Nature, Geography, History, Mythology, Food, Literature, etc.).
  - Direct links to corpus definitions and match confirmations.

---

## 🚀 Installation & Quick Start

### 1. Clone or Navigate to the Project Directory
```powershell
cd C:\Users\ADITYA\.gemini\antigravity\scratch\wordle_crossword_solver
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

Dependencies:
- `streamlit>=1.30.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`

### 3. Run the Unit Test Suite
```powershell
python test_solver.py
```

### 4. Launch the Web Application
```powershell
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 📁 Project Architecture

```
wordle_crossword_solver/
├── app.py                     # Streamlit frontend & interactive UI
├── wordle_engine.py           # Shannon entropy, candidate filtering & simulation logic
├── crossword_engine.py        # TF-IDF vectorization, regex parser & clue matcher
├── data/
│   ├── wordle_words.py        # Curated Wordle targets (2,305 words) & allowed dictionary
│   └── crossword_data.py      # Curated crossword clues, answers & dictionary definitions
├── requirements.txt           # Project dependencies
├── test_solver.py             # Automated unit tests
└── README.md                  # Documentation
```
