<img width="1536" height="1024" alt="db4a4a56-0eb0-44e3-813a-e332df541e91" src="https://github.com/user-attachments/assets/e2004dda-b014-44f9-bf02-b3f7352be2e8" />

# Recurrent Depth as a Testable Architecture Hypothesis: A NanoMythos Validation on FineWeb-Edu


This technical note reports a small-scale pretraining validation of a **Claude-Mythos-inspired recurrent-depth architecture**. The goal is not to claim that Claude uses this exact design. Instead, we use the open-source **OpenMythos** implementation as an architectural hypothesis, transplant its recurrent-depth idea into a **nanowhale / DeepSeek-V4-style small language model**, and test whether the resulting model improves held-out perplexity under a controlled pretraining setup.

The key result is that **T=1 recurrent depth consistently improves FineWeb-Edu 10K held-out perplexity across three random seeds**. At 5,000 training steps, the baseline model reaches an average PPL of **166.2**, while the T=1 recurrent model reaches **148.3**, producing an average gain of **17.9 PPL** and a **3/3 seed win rate**.

This makes T=1 a credible candidate for the next scale-up stage, especially toward a 100M–200M parameter pilot.

---

## 1. Background: From Mythos Speculation to Architecture Validation

Closed-source frontier models often inspire architectural speculation. One such hypothesis is the so-called **Claude Mythos** direction: a model architecture that may rely on some form of repeated internal computation, recurrent depth, or iterative refinement inside the Transformer stack.

The important point is that this type of claim is difficult to verify directly. We do not have access to Claude's internal architecture. Therefore, a more useful engineering question is not:

> Can we prove what Claude is using internally?

A better question is:

> If we implement a Claude-Mythos-inspired recurrent-depth mechanism in a controlled open model, does it improve pretraining efficiency?

That is the framing of this experiment.

We take the recurrent-depth structure suggested by **OpenMythos** and integrate the same high-level idea into a small DeepSeek-V4-style model family, referred to here as **NanoMythos**. The experiment then evaluates whether a single recurrent step, **T=1**, can improve language-modeling perplexity on FineWeb-Edu.

This turns a speculative architecture story into a measurable pretraining experiment.

---

## 2. Core Idea: Reusing Computation Instead of Only Adding Layers

Most Transformer scaling discussions focus on three familiar axes:

```text
More parameters
More data
More training compute
```

Recurrent-depth models introduce another axis:

```text
More repeated computation per token, with partial parameter reuse
```

Instead of only stacking more independent layers, a recurrent-depth architecture allows one or more internal blocks to be applied repeatedly. Conceptually, the model gets an additional opportunity to refine its hidden states before producing the next-token distribution.

A simplified architecture looks like this:

```text
Input tokens
   ↓
Token embedding
   ↓
Prelude Transformer blocks
   ↓
Recurrent / Mythos core repeated T times
   ↓
Coda Transformer blocks
   ↓
Language-model head
   ↓
Next-token prediction loss
```

In this experiment, we focus on the smallest non-trivial recurrent setting:

```text
T = 1
```

This is a deliberately conservative choice. T=1 is not meant to represent a dramatic multi-step reasoning loop. It is the minimal configuration that tells us whether the recurrent-path modification is directionally helpful under a fixed pretraining budget.

---

## 3. Model Setup: A DeepSeek-V4-Style Small Backbone with a Mythos Core

The host model is based on the **nanowhale** direction: a small, approximately 110M-scale language model using a DeepSeek-V4-style architecture. This makes it a practical testbed for architecture iteration because it is small enough for repeated experiments while still being closer to modern MoE-style design than a plain GPT-2 reproduction.

The validation compares two model variants:

| Variant | Description |
|---|---|
| **Baseline** | nanowhale-style small model without recurrent-depth insertion |
| **NanoMythos T=1** | same backbone family, with one recurrent pass through the Mythos-style core |

The goal is to isolate the practical impact of adding a single recurrent computation path. The experiment does not attempt to match the full parameter scale, training data, or production recipe of DeepSeek-V4 or Claude. It is an early architecture validation.

---

## 4. Experimental Design

The experiment is designed around one clean validation question:

> Under the same FineWeb-Edu 10K pretraining budget, does T=1 recurrent depth reduce held-out perplexity compared with the baseline?

### Dataset

The experiment uses **FineWeb-Edu 10K**, a compact educational-web pretraining subset. This dataset size is not intended to produce a strong final model. Its value is that it provides a fast, realistic enough pretraining distribution for early architecture validation.

### Training Budget

Each model is trained for:

```text
5,000 steps
```

For the main seed, checkpoints are also evaluated at intermediate stages:

```text
1,000 / 2,000 / 3,000 / 4,000 / 5,000 steps
```

This staged evaluation helps separate short-lived optimization artifacts from persistent training improvements.

### Random Seeds

To avoid over-interpreting a single lucky run, the experiment uses three random seeds:

```text
seed2025
seed2027
seed2048
```

A result is considered more meaningful if it survives this cross-seed check.

### Metric

The headline metric is held-out perplexity:

```text
PPL = exp(NLL)
```

Lower PPL indicates better next-token prediction on the held-out validation split.

---

## 5. Main Result: T=1 Improves PPL Across All Seeds

At 5,000 training steps, the T=1 recurrent model outperforms the baseline on every tested seed.

| Seed | Baseline PPL | T=1 PPL | Δ PPL, T=1 - Baseline |
|---:|---:|---:|---:|
| 2025 | 156.7 | 147.7 | -9.0 |
| 2027 | 187.7 | 151.4 | -36.3 |
| 2048 | 154.2 | 145.7 | -8.5 |
| **Mean** | **166.2** | **148.3** | **-17.9** |
| **Win rate** | — | — | **3/3** |

The result is notable for two reasons.

First, the improvement is not limited to a single seed. T=1 wins on **seed2025**, **seed2027**, and **seed2048**, giving a 3/3 cross-seed win rate.

Second, the seed2027 result is especially interesting. The baseline run is much weaker on this seed, reaching 187.7 PPL, while the T=1 model remains close to the other T=1 runs at 151.4 PPL. This suggests that the recurrent path may provide not only an average improvement, but also some robustness against seed-level instability.

That said, the robustness interpretation should still be treated as a working hypothesis. Three seeds are enough to justify a scale-up pilot, but not enough to make a broad statistical claim.

---

## 6. Training Dynamics: The Advantage Strengthens After 3,000 Steps

A common failure mode in architecture experiments is that a new module appears to help early in training but loses its advantage later. The staged evaluation on seed2025 does not show that pattern.

| Step | Baseline PPL | T=1 PPL | Δ PPL |
|---:|---:|---:|---:|
| 1,000 | 488.4 | 484.7 | -3.7 |
| 2,000 | 286.6 | 283.8 | -2.8 |
| 3,000 | 203.2 | 195.5 | -7.7 |
| 4,000 | 162.6 | 153.7 | -8.9 |
| 5,000 | 156.7 | 147.7 | -9.0 |

The T=1 model is slightly better at 1,000 and 2,000 steps, but the gap becomes clearer from 3,000 steps onward. By 5,000 steps, the improvement is still present.

This suggests that the recurrent-depth insertion is not merely an early optimization artifact. Instead, the benefit appears to become more visible as the model begins to fit the FineWeb-Edu distribution more effectively.

---

## 7. GPT-2 Standard as an External Reference

The report also includes a GPT-2 Standard / Small-style 124M external reference trained under the same FineWeb-Edu 10K short-budget setting.

| Model | Parameter Scale | Training Steps | Held-out PPL |
|---|---:|---:|---:|
| nanowhale T=1 | 110M | 5,000 | 147.7 |
| nanowhale baseline | 110M | 5,000 | 156.7 |
| GPT-2 Standard / Small | 124M | 5,000 | 273.5 |

This comparison should not be read as a pure architecture-only comparison. GPT-2 differs from nanowhale in tokenizer, block design, parameter organization, and training details.

However, it is still useful as an external reference point. Under the same short-budget FineWeb-Edu 10K setup, the nanowhale T=1 model is substantially ahead of the GPT-2 Standard reference. This supports the view that the current backbone plus recurrent-depth modification is worth scaling further.

A precise interpretation is:

> The NanoMythos T=1 setup shows stronger early training efficiency than the GPT-2 Standard reference under this specific 10K-data, 5K-step validation regime.

---

## 8. Length-Binned Analysis: The Gain Appears in the Main Validation Region

The validation set is concentrated mainly in the **200–500 character** range. In this dominant length bucket, T=1 again beats the baseline across all three seeds.

| Seed | Length Bucket | Δ PPL, T=1 - Baseline |
|---:|---|---:|
| 2025 | 200–500 chars | -9.0 |
| 2027 | 200–500 chars | -36.3 |
| 2048 | 200–500 chars | -8.5 |

This matters because it reduces the likelihood that the headline PPL improvement is driven by a small number of abnormal validation samples. The improvement appears in the main text-length region of the validation distribution.

A reasonable working hypothesis is that T=1 recurrent depth is particularly helpful for short-to-medium educational web text, where a single additional internal refinement pass can improve representation quality without destabilizing optimization.

---

## 9. Boundary Check: Why T=2 Is Not the Mainline Yet

The experiment also includes a T=2 boundary check on FineWeb-Edu 10K, seed2025, at 5,000 steps.

| Model | PPL @ 5,000 Steps |
|---|---:|
| nanowhale T=1 | 147.7 |
| nanowhale baseline | 156.7 |
| nanowhale T=2 | 283.6 |

T=2 performs significantly worse than both T=1 and the baseline under the current recipe.

This result should not be interpreted as a general rejection of deeper recurrence. A more careful interpretation is that **T=2 is not plug-and-play under the current training configuration**.

Possible causes include:

- the learning-rate schedule is not adapted to repeated-block computation;
- hidden-state drift may be amplified across recurrent passes;
- residual scaling or normalization may need to be redesigned;
- loop embeddings or step-aware gates may be required;
- the 5,000-step / 10K-data regime may be too small for T=2 to become useful.

Therefore, the current design-space decision is:

| Recurrent Depth | Status | Rationale |
|---|---|---|
| T=0 | Required baseline | Needed as the control group |
| T=1 | Main scale-up candidate | Wins 3/3 seeds and improves mean PPL |
| T=2 | Secondary research track | Current recipe underperforms and needs retuning |
| T≥4 | Deferred | Too costly and unstable for the current validation stage |

This is a useful narrowing of the search space. The mainline should focus on T=1 until larger-data validation is complete.

---

## 10. Engineering Smoke Test

Before scaling the experiment, the training path needs to be operationally reliable. The current report confirms that the single-model training pipeline passes the essential smoke checks.

| Check | Result |
|---|---|
| Training loss | decreases from roughly 23.6 to 13.8 |
| 200-step smoke run | completed successfully |
| Checkpoint saving | final checkpoint written correctly |
| Train / validation split | metadata and split IO verified |
| MoE + recurrent forward path | runs without model-path failure |
| Multi-process launch | deferred due to rendezvous / cluster networking configuration |

The multi-process issue is categorized as an infrastructure configuration problem rather than a modeling failure. The next engineering step is to fix the rendezvous / `MASTER_ADDR` / port setup and run a minimal two-device smoke test.

---

## 11. Interpretation: What the Result Does and Does Not Prove

The result supports a narrow but meaningful claim:

> A single recurrent-depth pass can improve early pretraining efficiency in a small DeepSeek-V4-style language model on FineWeb-Edu 10K.

The result does **not** prove that:

- Claude uses this exact architecture;
- recurrent depth is always better than adding layers;
- T=2 or deeper recurrence is invalid;
- the same gain will automatically transfer to billion-scale models;
- FineWeb-Edu 10K results are enough to claim production-level model quality.

The correct takeaway is more practical:

> T=1 recurrent depth has produced a stable enough signal to justify a larger validation run.

That is exactly the role of a good architecture pilot. It does not settle the whole research question, but it tells us where to invest the next round of compute.

---

## 12. Recommended Next Steps

### 12.1 Run FineWeb-Edu 50K

The first priority is to test whether the T=1 advantage survives a larger data regime.

Recommended setup:

```text
Dataset: FineWeb-Edu 50K
Models: baseline vs T=1 vs GPT-2 Standard
Steps: at least 5,000; preferably extend to 10,000
Seeds: at least 2–3 if budget allows
Metric: held-out NLL / PPL
```

If T=1 continues to win at 50K, the architecture signal becomes much stronger.

### 12.2 Start a 100M–200M T=1 Pilot

The current cross-seed result supports a 100M–200M parameter pilot with T=1 as the default recurrent-depth setting.

The goal should be clearly scoped:

```text
Validate whether the T=1 recurrent advantage persists
when the model scale approaches the GPT-2 Small / Standard range.
```

This run should not be positioned as a final model-quality benchmark. It is a scale-up validation of the architecture mechanism.

### 12.3 Keep T=2 as a Separate Ablation Track

T=2 should be studied separately with a modified recipe. Useful ablations include:

```text
lower learning rate
longer warmup
step-aware loop embeddings
residual damping
gated recurrent updates
stronger normalization
longer training schedule
larger dataset
```

The important principle is not to let T=2 consume the mainline scale-up budget until its optimization recipe is healthier.

### 12.4 Complete Distributed Training Smoke

Before large runs, the multi-process path should be validated:

```text
fix rendezvous configuration
run 2-device smoke
verify checkpoint consistency
measure throughput
compare single-device and multi-device loss curves
```

This is an engineering gate, not a scientific blocker.

---

## 13. Conclusion

This experiment provides a practical way to evaluate a Claude-Mythos-inspired architecture hypothesis without making unverifiable claims about closed-source systems.

By integrating an OpenMythos-style recurrent-depth core into a nanowhale / DeepSeek-V4-style small model, we can test whether recurrent computation improves pretraining behavior under a controlled setup.

The FineWeb-Edu 10K result is positive:

```text
Baseline mean PPL: 166.2
T=1 mean PPL:      148.3
Average gain:      -17.9 PPL
Seed win rate:     3/3
```

The most defensible conclusion is:

> T=1 recurrent depth is a credible architecture-improvement candidate for the next scale-up stage.

The next milestone is straightforward: run FineWeb-Edu 50K and a 100M–200M T=1 pilot. If the improvement persists under larger data and model scale, NanoMythos can move from a speculative architecture experiment to a more systematic pretraining research direction.
