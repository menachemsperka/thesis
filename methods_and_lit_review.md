# Split Strategy Results Summary

## Thesis Context

This document summarizes the methods and empirical results of the sentence-split study and its downstream impact on the thesis NER architectures.

The central research question was:

> Does a more label-aware sentence-level train/evaluation split improve Hebrew NER performance compared with a simple random split, and do those improvements transfer to more advanced downstream architectures?

The study was conducted in two stages:

1. **Direct split-strategy evaluation**: comparison of eight sentence-split strategies using a standard DictaBERT-based NER model.
2. **Downstream transfer evaluation**: reuse of the exact same saved split datasets in four additional architectures to test whether the split strategy improves more advanced NER systems.

## Methods

### Dataset and Task

- Task: Hebrew named entity recognition.
- Label format: token-level BIO labels with entity types.
- Splitting granularity: **sentence level**, to prevent token leakage between training and evaluation.
- Base train/evaluation ratio: **70% / 30%**.
- Reproducibility: the direct split-strategy evaluation used **5 seeds** (`42-46`) and reported mean and standard deviation.

### Sentence-Split Strategies Compared

Eight sentence-level splitting strategies were evaluated:

1. **Baseline (simple random split)**
2. **Label-aware greedy**
3. **Rare-label boosted**
4. **Inverse-frequency weighted**
5. **Min-max equalized**
6. **Inverse-frequency token-weighted**
7. **Inverse-frequency eval-guaranteed**
8. **Inverse-frequency log-scaled**

All strategies used the same underlying DictaBERT NER model and differed only in how sentences were assigned to training and evaluation.

### Rationale for the Split Strategies

- The **baseline** provides the control condition.
- **Label-aware greedy** tries to preserve non-`O` label distributions in train.
- **Rare-label boosted** prioritizes sentences containing rare labels.
- **Inverse-frequency weighted** explicitly favors rare-label-rich sentences.
- **Token-weighted inverse-frequency** strengthens that preference when rare labels appear multiple times in the same sentence.
- **Eval-guaranteed** preserves evaluation visibility for all labels.
- **Log-scaled inverse-frequency** softens extremely aggressive weighting while still prioritizing rare labels.

### Downstream Transfer Evaluation

After the direct split-strategy evaluation, the saved train/evaluation sentence splits were reused in four additional model settings:

- **AUC-2T**
- **AUC Cascaded Pipeline**
- **AUC Cascaded Pipeline with Step-3 consistency**
- **Fusion of regular NER and cascaded pipeline**

Each downstream architecture was rerun on the saved split datasets corresponding to all eight split variants. This isolates the effect of the split strategy from the effect of the model architecture.

## Results

### Results of the Direct Split-Strategy Evaluation

| Condition | F1 (mean ± std) | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| Baseline (simple random split) | 0.7228 ± 0.0278 | 0.6886 ± 0.0334 | 0.7608 ± 0.0244 | 0.9317 ± 0.0058 |
| Label-aware greedy | 0.7438 ± 0.0258 | 0.6923 ± 0.0339 | 0.8040 ± 0.0156 | 0.9460 ± 0.0066 |
| Rare-label boosted | 0.8078 ± 0.0259 | 0.7752 ± 0.0269 | 0.8438 ± 0.0331 | 0.9518 ± 0.0058 |
| Inverse-freq weighted | 0.8721 ± 0.0220 | 0.8315 ± 0.0291 | 0.9176 ± 0.0263 | 0.9737 ± 0.0042 |
| Min-max equalized | 0.7612 ± 0.0369 | 0.7258 ± 0.0506 | 0.8011 ± 0.0262 | 0.9521 ± 0.0054 |
| Inv-freq token-weighted | 0.8637 ± 0.0134 | 0.8202 ± 0.0330 | 0.9133 ± 0.0165 | 0.9733 ± 0.0034 |
| Inv-freq eval-guaranteed | 0.8195 ± 0.0129 | 0.7801 ± 0.0198 | 0.8638 ± 0.0232 | 0.9632 ± 0.0036 |
| Inv-freq log-scaled | **0.8782 ± 0.0108** | **0.8399 ± 0.0229** | **0.9212 ± 0.0245** | **0.9752 ± 0.0025** |

### Improvement Over the Random-Split Baseline

| Condition | Delta F1 | Delta Precision | Delta Recall | Delta Accuracy |
|---|---:|---:|---:|---:|
| Label-aware greedy | +0.0211 | +0.0037 | +0.0432 | +0.0143 |
| Rare-label boosted | +0.0850 | +0.0865 | +0.0830 | +0.0202 |
| Inverse-freq weighted | +0.1494 | +0.1429 | +0.1568 | +0.0420 |
| Min-max equalized | +0.0384 | +0.0372 | +0.0403 | +0.0204 |
| Inv-freq token-weighted | +0.1409 | +0.1316 | +0.1525 | +0.0416 |
| Inv-freq eval-guaranteed | +0.0968 | +0.0915 | +0.1030 | +0.0315 |
| Inv-freq log-scaled | **+0.1555** | **+0.1512** | **+0.1604** | **+0.0435** |

### Interpretation of the Direct Split-Strategy Evaluation

The results show a clear and consistent pattern:

- All label-aware strategies outperformed the simple random baseline.
- Strategies that explicitly prioritize rare-label coverage produced the largest gains.
- The strongest method was **inverse-frequency log-scaled**, which achieved the highest mean F1 and the lowest standard deviation among the top-performing methods.

This suggests that performance gains were not caused merely by randomness, but by better allocation of rare and informative entity-bearing sentences into the training set.

## Downstream Transfer Results

### Mean F1 Across the Downstream Architectures

| Variant | Mean F1 | Min F1 | Max F1 |
|---|---:|---:|---:|
| Inv-freq log-scaled | **0.7319** | 0.2321 | 0.9154 |
| Inverse-freq weighted | 0.6815 | 0.0609 | 0.9261 |
| Inv-freq token-weighted | 0.6706 | 0.0000 | 0.9011 |
| Inv-freq eval-guaranteed | 0.6568 | 0.1724 | 0.8452 |
| Rare-label boosted | 0.6022 | 0.1246 | 0.7927 |
| Baseline (simple random split) | 0.5611 | 0.1845 | 0.7082 |
| Label-aware greedy | 0.5559 | 0.2114 | 0.7228 |
| Min-max equalized | 0.5310 | 0.0240 | 0.7235 |

### Best Improvement Over Baseline by Architecture

| Architecture | Best observed delta vs baseline |
|---|---:|
| AUC-2T | **+0.0476** (Inv-freq log-scaled) |
| AUC Cascaded Pipeline | **+0.2156** (Inv-freq token-weighted) |
| AUC Cascaded Step-3 Consistency | **+0.2369** (Inv-freq log-scaled) |
| Fusion (Regular + Cascaded) | **+0.2179** (Inverse-freq weighted) |

### Full Delta Findings

#### AUC-2T

- Label-aware greedy: +0.0269
- Rare-label boosted: -0.0599
- Inverse-freq weighted: -0.1236
- Min-max equalized: -0.1605
- Inv-freq token-weighted: -0.1845
- Inv-freq eval-guaranteed: -0.0121
- Inv-freq log-scaled: **+0.0476**

#### AUC Cascaded Pipeline

- Label-aware greedy: -0.0066
- Rare-label boosted: +0.0454
- Inverse-freq weighted: +0.2005
- Min-max equalized: -0.0181
- Inv-freq token-weighted: **+0.2156**
- Inv-freq eval-guaranteed: +0.1345
- Inv-freq log-scaled: +0.1921

#### AUC Cascaded Step-3 Consistency

- Label-aware greedy: -0.0558
- Rare-label boosted: +0.0942
- Inverse-freq weighted: +0.1869
- Min-max equalized: +0.0450
- Inv-freq token-weighted: +0.2140
- Inv-freq eval-guaranteed: +0.1667
- Inv-freq log-scaled: **+0.2369**

#### Fusion (Regular + Cascaded)

- Label-aware greedy: +0.0146
- Rare-label boosted: +0.0845
- Inverse-freq weighted: **+0.2179**
- Min-max equalized: +0.0130
- Inv-freq token-weighted: +0.1928
- Inv-freq eval-guaranteed: +0.0934
- Inv-freq log-scaled: +0.2063

## Discussion

Three main findings emerge from these experiments.

### 1. Sentence splitting is not a neutral preprocessing choice

The direct split-strategy evaluation demonstrates that train/evaluation splitting alone can produce very large differences in NER performance. The difference between the baseline and the best strategy was **+0.1555 F1**, which is too large to be treated as a minor preprocessing artifact.

### 2. Rare-label-aware splitting transfers well to advanced pipelines

The downstream transfer study shows that the benefits of better split design are not limited to the regular DictaBERT model. The strongest downstream gains were seen in the cascaded and fusion systems, where improvements exceeded **+0.20 F1** in several conditions.

### 3. The best strategy depends on the model, but log-scaled inverse-frequency is the strongest overall choice

Although different downstream architectures had different single best variants, **inverse-frequency log-scaled** was the strongest overall strategy across the transfer study, with the highest mean F1 across all four downstream architectures (**0.7319**).

This makes it the most defensible general-purpose split strategy for the thesis pipeline.

## Conclusion

The results support the conclusion that **label-aware sentence splitting materially improves Hebrew NER performance**, especially in imbalanced datasets with rare entity types.

Among the tested approaches, **inverse-frequency log-scaled splitting** is the most robust overall method:

- it produced the best result in the direct split-strategy evaluation,
- it produced the best mean downstream performance across the four downstream architectures,
- and it consistently outperformed the simple random baseline.

Therefore, for the thesis experiments, the recommended train/evaluation split protocol is:

1. split at the **sentence level**,
2. preserve **rare-label coverage** in the training set,
3. use an **inverse-frequency log-scaled** sentence scoring rule when constructing the training split.

## Recommended Thesis Wording

One possible short summary sentence for the thesis is:

> A label-aware sentence split substantially improved Hebrew NER performance over a simple random split; the inverse-frequency log-scaled strategy was the strongest overall method, yielding the highest F1 in the direct split evaluation and the best average transfer performance across the downstream architectures.