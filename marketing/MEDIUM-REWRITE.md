# Medium rewrite — ready to paste

**What changed and why:** the original post opens with "I'm fascinated by real-time
time-series ML." That frames the most unfakeable thing about this project — that you are
the patient — as a generic ML exercise. It also repeats two claims that were false
(synthetic-only data, secrets never in Git), shows an API shape the code does not return,
and publishes no accuracy figures at all.

**Keep the original URL.** Edit in place. It already has indexing you cannot buy back.

**New title:** I Built a System to Predict My Own Blood Sugar. Then I Found Out It Missed
91% of the Events It Was Built For.

**Subtitle:** A Type 1 diabetic's cloud-native CGM predictor — and what a proper
evaluation revealed a year later.

---

> **Not a medical device.** Research and education only. Never use this to make treatment
> decisions.

I have Type 1 diabetes. I built this for myself, in my first semesters of university,
before I had any professional reason to build anything.

If you use a continuous glucose monitor, you know the trade. Set your low alarm at 90 and
it screams all day until you stop hearing it. Set it at 80 and by the time it fires with
one arrow down, you are going to 65 whatever you do — and that costs you the next hour.
Neither setting gives you a decision you can act on.

The reason is pharmacokinetics, not software. Fast carbohydrate takes 15–20 minutes to
reach your blood. Corrective insulin takes considerably longer. By the time the arrow
turns down far enough to alarm you, the intervention you make in response is already
fifteen minutes late.

So I tried to move knowing fifteen minutes earlier.

## What I built

BSP is a per-user glucose forecaster. Every user gets their own trained models, because an
early experiment training on four other people's traces produced visibly worse results
than training on my own. Glucose response is idiosyncratic enough that a population model
gave up more than it gained.

The deployed stack is fully serverless:

- **App Runner** — headless export of 90 days of CGM history into S3
- **SageMaker Processing** — trains a bag of LSTM models, scores them, exports to ONNX
- **Lambda** — loads the bundle and returns a 15-minute forecast, scale-to-zero
- **S3** — per-user model bundles and training data

The model reads twelve 5-minute readings — one hour — with time-of-day encoded as sine and
cosine, and predicts one step ahead. Three recursive steps give the 15-minute horizon.
Training runs in Keras; inference runs on ONNX, which keeps the Lambda image small enough
to cold-start in about 40 seconds.

Two choices mattered more than I expected. Splitting training and inference into separate
images means the serving path never carries TensorFlow. And time-of-day turned out to be
load-bearing: re-running my evaluation with a frozen clock instead of a real advancing one
nearly doubles error at the 15-minute horizon, from 9.3 to 17.0 mg/dL.

There are working clients — a React dashboard and a Kotlin/Compose Android app — showing
the last twelve readings, the next three predictions, a confidence gauge, and anomaly
detection.

## Then both doors closed

The patent application was rejected. Google Play blocked the app on medical-software
requirements, after I had paid the developer subscription, written a privacy policy, and
completed the submission.

I open-sourced everything in September 2025 and set it down. Then I left it alone for
almost a year.

## Coming back to it, and what I found

When I returned in August 2026 I did the thing I had never properly done: I evaluated it.
Held-out split, real baselines, event-level metrics.

The average-error numbers looked fine. At the 15-minute horizon, mean absolute error of
9.3 mg/dL, beating both a persistence baseline (10.7) and linear extrapolation (14.7). I
included persistence deliberately — a glucose predictor that cannot beat *assume the
current value holds* is not predicting anything, and most write-ups quietly skip that bar.

Then I split the test set by what the glucose actually did.

| Window type | n | MAE |
|---|---|---|
| all windows | 830 | 8.9 |
| quiet (±10 mg/dL) | 542 | **5.1** |
| moving (>20 mg/dL) | 99 | **26.8** |
| falling fast (< −20) | 45 | **28.6** |

**Sixty-five percent of CGM windows barely move.** My aggregate error was carried almost
entirely by them. On the windows that actually matter, error was five times worse.

And the number that stopped me:

> **Hypoglycemia detection at 15 minutes: 3 events caught out of 34. Recall 8.8%.**

The system I designed, deployed, ran for months, and wrote a public article about caught
fewer than one in ten of the events it existed to catch.

Here is one of them. On 3 May 2025 at 21:12 I was at 156 mg/dL and felt fine. Twenty
minutes later I was at 69. My deployed model predicted 152.

## Why it happened

Trained with mean squared error on a signal that is flat two thirds of the time, the
network had learned that predicting *roughly the same as now* minimises loss.

That is the loss function working perfectly on a badly posed objective. My model had
quietly become an expensive persistence predictor — and because persistence scores well on
average error, no metric I was watching could reveal it.

## The fix that cost nothing

Before retraining anything, I checked whether individual models were better than their
average. They were, dramatically:

| Strategy | Caught | Recall | Precision |
|---|---|---|---|
| ensemble **mean** — what I deployed | 3 / 34 | **8.8%** | 33% |
| best single model | 16 / 34 | 47.1% | 50% |
| ensemble **minimum** | 17 / 34 | **50.0%** | 50% |

**Averaging was destroying the signal.** When two models in a bag of ten forecast a fall
to 65 and eight forecast 150, the mean lands near 133 and nothing fires.

Averaging is the right operation for a point estimate and the wrong one for a warning,
because a warning is a claim about the worst plausible outcome, not the consensus one. My
bag of models was working. The reduction step was throwing the answer away.

Same models, same weights, no retraining — just a different way of counting the votes —
and recall went from 8.8% to 50%.

This is the part I think generalises past glucose. If you ensemble models to detect rare
events and you average their outputs, you are averaging away the minority that spotted the
event. That applies to fraud, to fault detection, to anything where the interesting cases
are rare.

## And then retraining

Aggregation is a patch; the objective was the real problem. So I retrained with losses that
stop rewarding the flatline — an asymmetric loss penalising a missed fall more heavily than
a false alarm, and a variant predicting the low-glucose event directly as a classification
instead of regressing a value and thresholding it.

| Objective | Event | Recall | Precision |
|---|---|---|---|
| MSE (as deployed) | <70 mg/dL | 8.8% | 33.0% |
| **asymmetric loss** | <70 mg/dL | **73.8%** | 43.7% |
| asymmetric, stronger | <70 mg/dL | 88.3% | 22.9% |
| **event classifier** | <80 mg/dL | **75.7%** | 66.3% |

On the identical metric, recall went from 8.8% to 73.8% — an 8.4-fold improvement, with
precision rising too.

## What this does not show

All of it is one person. 24,947 readings across 90 days, all mine. The 34 hypoglycemic
events in the test window are enough to demonstrate an eightfold gap; they are nowhere near
enough to tune a clinical threshold or claim the approach transfers. There is no trial, no
ethics approval, no clinical validation. At 43.7% precision more than half the asymmetric
model's warnings are false, and whether that trade is acceptable is a question about human
behaviour I have not measured. The retrained models are not deployed — those figures are
offline. And the Dexcom integration still relays an account password on every call, because
it wraps an unofficial API; a real build needs the official OAuth path.

I would rather publish those five sentences than a number without a denominator.

## Corrections to the original version of this post

In the interest of the same standard: the earlier version of this article said the
repository ships synthetic demo data only. It does not — it ships 24,947 of my own real CGM
readings, published deliberately. It also said secrets were kept out of Git, which was not
true at the time; a hard-coded credential sat in the repository and has since been rotated
and purged from history. The `/predict` response shape shown previously was aspirational
rather than what the code returns.

## What I actually learned

**Engineering is not product.** A working system can be stopped dead by distribution rules
that have nothing to do with whether it works.

**Average error is the wrong instrument for a warning system.** It is the metric that is
easiest to compute and the one most likely to flatter you.

**Not evaluating properly is a decision.** I shipped, wrote about it, and moved on without
ever measuring the thing the system existed to do. The bug was live in my production UI for
months, printing `Trend Change: 0.000` on every model, and I never asked why.

**Publishing the failure is worth more than hiding it.** The 8.4× improvement only means
something because the starting point is stated honestly.

---

**Full technical report with methodology:** [link to the artifact]
**Code:** https://github.com/Atakan-Kaya35/BSP
**Web client:** https://github.com/Atakan-Kaya35/blood-sugar-predictor-web
**Android client:** https://github.com/Atakan-Kaya35/blood-sugar-predictor-android

*Not a medical device. Research and education only.*
