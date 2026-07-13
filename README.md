# churn-uplift

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akshay-birla-03/churn-uplift/blob/main/notebooks/Run_in_Colab.ipynb)

**Causal / uplift modeling toolkit for retention marketing.**

Most churn projects predict *who will leave*. That is the wrong question for a
retention campaign. If you give a discount to a customer who was going to stay
anyway, you burned margin. If you contact a customer who resents being nudged,
you *caused* the churn you were trying to prevent. What you actually want is the
**incremental effect of the treatment per customer** — the *uplift* — so you can
spend your budget only on the people the offer genuinely moves.

`churn-uplift` (import name `upliftkit`) implements two meta-learners
(**S-learner**, **T-learner**) over any scikit-learn classifier and evaluates
them with **uplift-specific** metrics (Qini curve & coefficient, uplift@k,
cumulative gain / AUUC). It ships a synthetic-but-realistic telecom retention
generator with a **known heterogeneous treatment effect**, so the test suite can
assert the models genuinely *recover* that effect rather than just running.

Everything runs **fully offline**. No data downloads, no network.

---

## The four customer segments

Uplift modeling divides customers by how a retention treatment changes their
behavior — not by whether they churn:

```
                     Retained WITHOUT treatment?
                        NO              YES
                  +---------------+---------------+
     Retained     |  PERSUADABLE  |  SURE THING   |
     WITH     YES |  treat them!  | don't bother  |
     treatment?   |  (uplift +)   |  (uplift ~0)  |
                  +---------------+---------------+
              NO  |  LOST CAUSE   | SLEEPING DOG  |
                  | don't bother  |  DO NOT TREAT |
                  |  (uplift ~0)  |  (uplift < 0) |
                  +---------------+---------------+
```

* **Persuadable** — retained *only if* treated. Large positive uplift. Target these.
* **Sure thing** — loyal regardless. ~zero uplift. Treating them wastes budget.
* **Lost cause** — churns regardless. ~zero uplift. Unreachable.
* **Sleeping dog** — the outreach *annoys* them; treatment lowers retention.
  **Negative uplift.** Targeting them is actively harmful.

The synthetic generator (`upliftkit.data.generate`) bakes these segments in with
fixed baseline and effect probabilities, randomizes treatment as a true RCT
(independent of features), and records each row's **true uplift** for validation.

---

## S-learner vs T-learner

Neither the treated-world nor the control-world outcome is ever observed for the
same customer, so uplift is estimated by modeling both potential outcomes.

* **S-learner** ("single"): one model on `[features, treatment_flag]`. Predict
  with the flag set to 1 and to 0; uplift is the difference. Simple, data-efficient,
  but can under-fit the treatment interaction if the base learner regularizes the
  flag away.
* **T-learner** ("two"): a separate model per arm (treated / control). Captures
  arm-specific structure well, but each model sees only half the data and their
  errors can compound.

Both wrap any classifier with `predict_proba` (default:
`GradientBoostingClassifier`; pass `base_estimator=LogisticRegression()` etc.).

---

## The Qini metric

Rank customers by predicted uplift, then target from the top down. At each prefix
the **Qini curve** plots incremental positive outcomes captured, correcting for
unequal treated/control counts:

```
g(n) = Y_treated(n) - Y_control(n) * N_treated(n) / N_control(n)
```

A good model's curve bows above the diagonal (random targeting). The **Qini
coefficient** here is the area between the curve and that diagonal, normalized by
population size — positive beats random, ~0 is useless. `uplift@k` reports the
observed treated-minus-control response gap within the top-`k` fraction.

---

## Measured results

From a single reproducible run (`upliftkit`, 4000 synthetic customers, seed 7,
30% held out — reproduce with the CLI below):

| Learner   | Qini coefficient | uplift@10% | corr(pred, **true** uplift) |
|-----------|:----------------:|:----------:|:---------------------------:|
| S-learner |     +0.0268      |   +0.529   |          **+0.953**         |
| T-learner |     +0.0275      |   +0.435   |          **+0.794**         |

Both learners recover the heterogeneous effect: predicted uplift correlates
strongly with the *known* per-customer ground truth (Pearson r of **0.95** and
**0.79**), the Qini coefficient is positive, and the top-decile realized uplift
(~0.4–0.5) is far above the ~0.15 average treatment effect — exactly the
concentration of persuadables you want. Numbers are honest outputs of the code in
this repo, not hand-picked; the test suite enforces r ≥ 0.3 and Qini > 0.

---

## Install

```bash
pip install -e ".[dev]"      # or: make install
```

Only depends on numpy, pandas, scikit-learn, joblib.

## Usage

CLI:

```bash
upliftkit                       # default run
upliftkit -n 8000 --seed 3 --test-size 0.25
```

Library:

```python
from upliftkit import generate, SLearner, TLearner, evaluate_learner
from sklearn.model_selection import train_test_split

ds = generate(n=4000, seed=7)
train, test = train_test_split(ds.frame, test_size=0.3, random_state=7,
                               stratify=ds.frame["treatment"])
cols = ds.feature_columns

model = TLearner().fit(train[cols], train["treatment"], train["outcome"])
uplift = model.predict_uplift(test[cols])          # per-customer incremental effect

res = evaluate_learner(model, test[cols], test["treatment"], test["outcome"],
                       true_uplift=test["true_uplift"])
print(res.qini_coefficient, res.true_uplift_correlation)
```

Or run the whole thing end to end:

```python
from upliftkit import run_pipeline
result = run_pipeline(n=4000, seed=7)
print(result.best_learner, result.best.qini_coefficient)
```

## Testing

```bash
make test        # python -m pytest -q
make lint        # ruff check src tests
```

The suite (38 tests) covers: metric correctness on hand-computed toy examples,
generator determinism and per-segment effect signs, empirical-vs-true ATE
recovery, learner fit/predict contracts, and the quality bar (predicted-vs-true
uplift correlation ≥ 0.3 with positive Qini on held-out data). Full run < ~15s.

## Project layout

```
src/upliftkit/
  data.py       synthetic RCT generator + known per-row uplift
  learners.py   SLearner, TLearner meta-learners
  metrics.py    qini_curve, qini_coefficient, uplift_at_k, cumulative_uplift_curve, auuc
  evaluate.py   evaluate_learner -> EvaluationResult dataclass
  pipeline.py   generate -> split -> fit S & T -> evaluate -> pick best
  cli.py        `upliftkit` entry point
tests/          pytest suite
```

## License

MIT — see [LICENSE](LICENSE).
