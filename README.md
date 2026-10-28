# Indian Cricket Match Analysis (IPL 2008–2025)

Answering cricket questions with data instead of opinion: exploratory analysis of every IPL match from 2008 to 2025, plus machine-learning models that predict **who wins**, **how many runs are scored**, and **how a chase unfolds after 6, 10 and 15 overs**.

> Project 14 – *Indian Cricket Match Analysis* · Group 5 · IDC 409

---

## Headline results

| Question | Answer |
|---|---|
| Does the toss matter? | Barely. Toss winners win **50.6%** of matches. Choosing to field wins **53.4%** vs **45.2%** for choosing to bat. |
| Bat first or chase? | Chasing wins more: **615** chase wins vs **531** batting-first wins (23 tied or no result). |
| Is there home advantage? | Yes for some teams: CSK 66% home vs 52% away, MI 61% vs 51%, RR 61% vs 44%, KKR 57% vs 48%. |
| Which phase scores fastest? | Death overs: **9.52** runs/over vs 7.66 (powerplay) and 7.63 (middle). Powerplay has the most dot balls (44.8%). |
| Can we predict the winner before the match? | **No reliable edge.** 49–52% accuracy vs a 50–53% majority baseline. |
| Can we predict a chase while it is happening? | **Yes.** Accuracy rises from **76% → 78% → 86%** at 6 / 10 / 15 overs (ROC-AUC 0.85 → 0.87 → 0.93), beating the required-run-rate rule. |

---

## Dataset

- **Source:** Kaggle – [IPL Dataset 2008–2025 (`chaitu20/ipl-dataset2008-2025`)](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025), file `IPL.csv`, downloaded with `kagglehub`.
- **Size:** 278,205 ball-by-ball rows × 64 columns, covering **1,169 matches** and 18 seasons.
- **Cleaning:** franchise renames merged (Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab Kings, Deccan Chargers → Sunrisers Hyderabad, and so on); venue spellings merged down to **36** grounds; missing winners, outcomes and cities filled; dates parsed.

---

## Repository contents

| Notebook | Author tag | What it does |
|---|---|---|
| `Match_Analysis__MS23238_.ipynb` | MS23238 | Cleans the data into a match table (`data/cleaned_matches.csv`) and answers the match-level EDA questions (toss, batting first vs chasing, venues, home advantage, awards, DLS, scoring ranges). |
| `Ball_by_ball_analysis_MS23131_.ipynb` | MS23131 | Ball-by-ball analysis: phase dynamics, batter–bowler matchups, extras, wickets and boundaries by over, innings totals, batter profiles. |
| `ML-_FIRST_INNINGS.ipynb` | – | Pre-match winner classification, first-innings score regression, live first-innings projection, venue clustering, SHAP, and the corrected leak-free pipeline. |
| `ML-2ND_INNINGS.ipynb` | – | Chase win-probability and chase-total models at 6 / 10 / 15 overs, what-if heatmaps, a live predictor and a dashboard. |

Suggested layout (adjust to match the repo):

```
├── README.md
├── IPL_Match_Analysis.pptx
├── data/                      # cleaned_matches.csv is written here
├── results/                   # CSV outputs and dashboard.png written by the ML notebooks
└── notebooks/
    ├── Match_Analysis__MS23238_.ipynb
    ├── Ball_by_ball_analysis_MS23131_.ipynb
    ├── ML-_FIRST_INNINGS.ipynb
    └── ML-2ND_INNINGS.ipynb
```

---

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub shap
```

Run the notebooks in this order:

1. `Match_Analysis__MS23238_` – downloads the data and writes `data/cleaned_matches.csv`
2. `Ball_by_ball_analysis_MS23131_`
3. `ML-_FIRST_INNINGS`
4. `ML-2ND_INNINGS`

**Path note:** a few cells still contain hard-coded Windows paths (for example `C:\Users\...\IPL.csv`). Replace them with the path returned by `kagglehub.dataset_download(...)`, or set `RAW_CSV` in the first-innings notebook.

---

## Exploratory analysis

1. **Toss impact:** 50.56% of toss winners won the match; 65% of captains chose to field.
2. **Toss decision:** fielding first wins 53.4% of the time for the toss winner, batting first only 45.2%.
3. **Batting first vs chasing:** 531 vs 615 wins; in the 22 DLS-affected matches the side batting first won 12 and the chasing side 10.
4. **Venue effect:** most matches at Wankhede (Mumbai Indians dominant, 56 wins), Eden Gardens (KKR, 54), Chepauk (CSK, 51), Chinnaswamy (RCB, 45) and Arun Jaitley (DC, 36).
5. **Home advantage:** see the table below.
6. **Team trends and awards:** season-by-season win trends for the big four franchises; Chris Gayle leads the single-season Player-of-the-Match count (6 in 2011).
7. **Phase dynamics:** run rate, wickets and dot-ball % by powerplay / middle / death.
8. **Scoring range:** first-innings mean **167** runs (median 167, middle half 147–188, range 56–287).
9. **Matchups:** highest-scoring pairs include Warner vs Narine (195), Kohli vs Ashwin (181), Kohli vs Jadeja (179), Raina vs Chawla (175), Rohit vs Umesh Yadav (170).

| Team | Home win % | Away / neutral win % |
|---|---|---|
| Chennai Super Kings | 66.2 | 52.0 |
| Mumbai Indians | 60.9 | 51.4 |
| Rajasthan Royals | 61.3 | 43.9 |
| Kolkata Knight Riders | 56.8 | 47.9 |
| Sunrisers Hyderabad | 49.4 | 43.2 |
| Punjab Kings | 47.9 | 44.0 |
| Royal Challengers Bangalore | 47.4 | 49.7 |
| Delhi Capitals | 41.4 | 45.6 |

*Home grounds are matched by name prefix. A later cell in the notebook matches on the first word only, which over-counts (for example 248 "home" matches for RCB), so the table above uses the stricter version.*

---

## Modelling

All final models use **chronological evaluation** (walk-forward by season in the first-innings notebook, an 80/20 time-ordered split in the second-innings notebook), and every model is compared with a baseline. Features are built only from matches played *before* the one being predicted.

### Pre-match match-winner classification (test seasons 2018–2025, n = 520)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Baseline: majority class | 53.1% | 0.483 |
| Logistic Regression | 51.5% | 0.479 |
| Random Forest | 49.2% | 0.488 |
| Gradient Boosting | 50.2% | 0.504 |

Features: Elo difference, recent form, head-to-head, venue average score, venue batting-first rate, toss winner. **None beats the baseline** (the baseline sits inside every model's 95% confidence interval), so pre-match outcomes look close to a coin flip with this feature set.

### Pre-match first-innings score regression (n = 516)

| Model | MAE (runs) | RMSE | R² |
|---|---|---|---|
| Baseline: train-set mean | 27.9 | 35.8 | −0.14 |
| Baseline: venue average | 26.8 | 34.7 | −0.07 |
| Ridge (best) | **26.3** | 33.5 | 0.00 |

### Live prediction after 6, 10 and 15 overs

**Chase outcome (chasing team wins?)**, second-innings notebook:

| Checkpoint | Majority baseline | Required-rate rule | Best model | ROC-AUC |
|---|---|---|---|---|
| Pre-match | 49.8% | – | 51.6% (Gradient Boosting) | 0.507 |
| 6 overs | 49.8% | 70.7% | **76.4%** (Logistic Regression) | 0.846 |
| 10 overs | 48.9% | 71.3% | **77.6%** (Logistic Regression) | 0.873 |
| 15 overs | 47.9% | 78.0% | **86.1%** (Logistic Regression) | 0.934 |

Models were chosen by time-series cross-validated AUC, not by test-set score. Features: runs, wickets, last-5-over runs, current and required run rate, runs needed, target, venue chase history and recent form.

**Score regression (MAE in runs)**:

| Checkpoint | 1st-innings projection (Gradient Boosting) | Baseline: run rate × 20 | Chase total (Gradient Boosting) | Baseline: run rate × 20 |
|---|---|---|---|---|
| 6 overs | 21.0 | 28.9 | 14.9 | 34.8 |
| 10 overs | 17.2 | 22.0 | 13.4 | 23.1 |
| 15 overs | 11.0 | 14.9 | 9.5 | 15.6 |

**Live predictor example** (`predict_live_chase`): chasing 191, the model gives 36% after 6 overs (48/1), 30% after 10 (85/3) and 8% after 15 (120/5); chasing 151 it gives 87%, 97% and 99% from 55/1, 100/1 and 128/2.

### Venue archetypes (K-Means, k = 3, venues with ≥ 10 matches)

| Archetype | Avg 1st-innings score | Chase win rate |
|---|---|---|
| High scoring | 187 | 52% |
| Balanced | 175 | 46% |
| Low scoring | 161 | 55% |

Silhouette scores favoured k = 3 (0.43).

### Explainability

SHAP summary and waterfall plots are provided for the pre-match model and for the 6 / 10 / 15-over chase models.

---

## What we corrected along the way

The notebooks keep the early attempts and then a *Corrections and final modifications* section. In short:

- The first four classification models and the first regression used `cleaned_matches.csv`, which holds **one row per match (the first delivery only)**. "Powerplay" features were really first-ball values, and the regression MAE of 1.06 predicted the first-ball running total. Those headline numbers (55.6% accuracy, MAE 1.06, R² 0.98 for the early chase regression) are **not valid** and are superseded by the leak-free results above.
- The match-winner target is now "does the team batting first win?", with both teams' strengths used as features (the early predictor ignored the two team names).
- Matches with no result, D/L adjustments and super overs are excluded from the chase models; train/test splits are chronological; model choice uses cross-validation.

---

## Limitations and next steps

- Pre-match features (toss, Elo, form, venue) carry little signal; adding **player-level** information (squads, bowler types, injuries) is the obvious next step.
- Live models use only score state; pitch, dew and day/night effects are not modelled.
- Test sets are modest (about 210–520 matches), so differences of a few points are within noise.
- The venue-level win-rate and fortress features should be recomputed strictly from past matches wherever they are still used in older cells.

---

## Deliverables checklist (from the project brief)

- [x] Clean match table and ball-by-ball aggregation
- [x] EDA notebook answering 6–8+ cricket questions
- [x] Model comparison against baselines (classification: accuracy, F1, confusion matrix; regression: MAE / RMSE)
- [x] Extension: live-style prediction after 6, 10 and 15 overs
- [x] Dashboard-style summary and this README
- [x] Presentation: `IPL_Match_Analysis.pptx`

---

## Team

Group 5 – Satyabrata Patra (MS23131) and Sohamita Ghosal (MS23238).

## Acknowledgements

Data from Kaggle (`chaitu20/ipl-dataset2008-2025`).
