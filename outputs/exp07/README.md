# Experiment 7 Output Guide

## What This Folder Is

This folder contains the outputs of the sentence-split study.

The purpose of this study is simple:

> Instead of changing the model, change **how sentences are divided into training and evaluation sets**, and check whether that alone improves named entity recognition performance.

The model itself stays the same. What changes is the rule used to decide which sentences go into training and which sentences stay for evaluation.

This matters because a simple random split can accidentally leave too many rare labels out of the training set. When that happens, the model does not see enough examples of those labels during learning, and performance suffers.

## What Is Happening Here

The study compares **8 different sentence-splitting strategies**.

Each strategy was tested with **5 seeds**.

For each run:

1. The dataset is split at the **sentence level**.
2. The model is trained on the training sentences.
3. The model is evaluated on the held-out sentences.
4. F1, precision, recall, and accuracy are recorded.
5. The results are averaged across the 5 seeds.

So this folder is not mainly about model architecture. It is about the effect of **data partitioning strategy**.

## Detailed Explanation of Each Variant

### 1. Baseline

- **Internal name:** `before_exp01_baseline`
- **Plain meaning:** sentences are shuffled randomly, and then the first 70% are used for training.

This is the simplest possible approach. It does not look at labels at all. A sentence containing a very rare entity type is treated exactly the same as a sentence containing only common labels.

This method is useful because it gives the reference point for the whole study. If a smarter split is genuinely useful, it should beat this baseline.

The weakness of this method is that random assignment can accidentally create a poor training set. In particular, rare labels may end up appearing mostly in evaluation instead of training, which means the model is asked to predict labels it barely saw while learning.

### 2. Label-aware greedy

- **Internal name:** `after_label_aware_split`
- **Plain meaning:** the split tries to keep the label distribution in training close to the label distribution of the full dataset.

This method is the first step beyond random splitting. Instead of assigning sentences blindly, it examines which labels occur in each sentence and tries to build a training set whose non-`O` label counts roughly match the full corpus.

The word **greedy** means it builds the training set step by step. At each step it chooses a sentence that seems to improve the label balance the most at that moment.

Why this helps:

- it reduces the chance that training becomes badly unbalanced,
- it tries to preserve the overall structure of the dataset,
- and it helps prevent rare labels from disappearing entirely from training.

Why it is still limited:

- it mainly tries to match the general label distribution,
- it does not strongly emphasize the rarest labels,
- and it may still behave too conservatively when some labels are extremely uncommon.

This explains why it improves on the baseline, but only modestly.

### 3. Rare-label boosted

- **Internal name:** `after_rare_boosted`
- **Plain meaning:** every sentence that contains a rare label is pushed into the training set first.

This method is much more direct than the label-aware greedy approach. Instead of trying to preserve the whole distribution softly, it says: if a sentence contains a rare label, it is especially valuable for learning and should be prioritized for training.

The practical effect is that the training set becomes much richer in the labels that the model is most likely to miss under random splitting.

Why this helps:

- it aggressively protects rare labels,
- it increases the chance that the model sees those labels during training,
- and it reduces the risk of evaluation containing labels that the model barely learned.

Why it is not always optimal:

- it treats all rare-label sentences as high priority,
- but it does not distinguish strongly between a sentence that contains one rare token and a sentence that contains many,
- and it may overfill training with rare-label sentences without optimizing the balance among them.

This makes it clearly stronger than baseline, but still less refined than the best inverse-frequency methods.

### 4. Inverse-frequency weighted

- **Internal name:** `after_inverse_freq_weighted`
- **Plain meaning:** labels that are rarer in the dataset receive larger weights, and sentences are scored based on how many rare labels they contain.

This is one of the strongest ideas in the study. Instead of just saying "rare labels matter," it gives each label a weight based on how uncommon it is. A sentence containing a very rare label therefore becomes much more valuable than a sentence containing only frequent labels.

The key intuition is simple:

- common labels are already easy to see many times,
- rare labels are hard to learn unless the split actively protects them,
- so the split should give more importance to rare labels than to frequent ones.

Why this works well:

- it focuses directly on the main weakness of random splitting,
- it gives a principled advantage to rare-label-rich sentences,
- and it improves training exposure where the model needs it most.

Why it can still be imperfect:

- very rare labels may get extremely large weights,
- which can make the split overly aggressive,
- and in some datasets that can lead to instability or overemphasis on a very small number of sentences.

Even with that limitation, this strategy performed extremely well.

### 5. Min-max equalized

- **Internal name:** `after_minmax_equalized`
- **Plain meaning:** tries to improve the worst-covered label first, so that no label is left too weakly represented in training.

This method is driven by fairness across labels. Instead of optimizing total label score, it asks: which label is currently in the worst situation, and which sentence would help that label the most?

The goal is to make the training set more even across entity types.

Why this idea is attractive:

- it tries to prevent any one label from being ignored,
- it explicitly protects the weakest-covered class,
- and it can improve robustness when the dataset is strongly imbalanced.

Why it did not perform as well as the top methods:

- it focuses on the minimum coverage ratio,
- which may help one weak label at a time,
- but it may not be as effective at building an overall high-information training set.

So it is conceptually reasonable, but in practice it was not competitive with the inverse-frequency family.

### 6. Inverse-frequency token-weighted

- **Internal name:** `after_inverse_freq_token_weighted`
- **Plain meaning:** similar to inverse-frequency weighting, but also rewards sentences that contain many rare-label tokens, not just the presence of a rare label.

This is a refinement of inverse-frequency weighting. The previous method looks mainly at whether a rare label appears in a sentence. This method goes further and asks how much rare-label content the sentence actually contains.

That means:

- a sentence with one rare token is useful,
- but a sentence with several rare tokens is even more valuable,
- so the split should prefer the second sentence more strongly.

Why this is a strong approach:

- it captures both rarity and density of useful information,
- it makes better use of sentences that are especially rich in training signal,
- and it avoids treating all rare-label sentences as equally informative.

Why it is not always the absolute best:

- it can become slightly too concentrated on a smaller set of dense rare-label sentences,
- which may reduce some aspects of balance,
- but overall it remained one of the top-performing methods.

### 7. Inverse-frequency eval-guaranteed

- **Internal name:** `after_inverse_freq_eval_guaranteed`
- **Plain meaning:** first guarantees that every label appears in evaluation at least once, then fills the training set using inverse-frequency logic.

This method tries to solve two problems at the same time:

1. the training set should contain enough rare labels to learn from,
2. the evaluation set should still contain those labels so performance can be meaningfully measured.

In other words, it tries to avoid a situation where evaluation becomes uninformative because some labels disappear from it completely.

Why this is useful:

- it makes evaluation more interpretable,
- it ensures that all labels are still being tested,
- and it keeps many of the benefits of rare-label-aware training.

Why it was not as strong as the best training-focused methods:

- reserving sentences for evaluation means some useful rare-label sentences are intentionally kept out of training,
- so there is a tradeoff between better measurement and maximum training benefit.

This explains why it performed well, but not at the level of the best inverse-frequency training-first approaches.

### 8. Inverse-frequency log-scaled

- **Internal name:** `after_inverse_freq_log_scaled`
- **Plain meaning:** still favors rare labels, but reduces the extreme effect of very rare labels by using a softer weighting rule.

This method keeps the central insight of inverse-frequency weighting, namely that rare labels should matter more than common ones. However, instead of letting that weighting become too extreme, it applies a logarithmic transformation.

In practical terms, this means:

- rare labels still get priority,
- but not so aggressively that a few extremely rare cases dominate the whole split,
- which produces a better balance between rare-label protection and overall stability.

Why this was the best method in the study:

- it preserves the main advantage of rare-label-aware splitting,
- it reduces the instability caused by extreme weighting,
- and it produces both high performance and low variance across seeds.

This makes it the most balanced strategy overall: strong rare-label coverage, strong training usefulness, and stable results.

## Main Results

The table below shows the average F1 results across 5 seeds.

| Variant | Description | Seeds | Mean F1 | F1 Std |
|---|---|---:|---:|---:|
| after_inverse_freq_log_scaled | Log-scaled inverse-freq: log(1+max/count) dampens extreme rare-label weights | 5 | **0.878248018** | 0.010825350 |
| after_inverse_freq_weighted | Inverse-frequency weighted: rare-label-rich sentences prioritized for train | 5 | 0.872148840 | 0.022048604 |
| after_inverse_freq_token_weighted | Inverse-freq token-weighted: score by token counts not just label presence | 5 | 0.863713471 | 0.013354927 |
| after_inverse_freq_eval_guaranteed | Inverse-freq eval-guaranteed: reserve 1 sentence per label for eval first | 5 | 0.819542789 | 0.012892009 |
| after_rare_boosted | Rare-label boosted: all sentences with rare labels forced into train first | 5 | 0.807820206 | 0.025873392 |
| after_minmax_equalized | Min-max equalized: greedily maximize minimum per-label coverage ratio in train | 5 | 0.761208189 | 0.036918985 |
| after_label_aware_split | Statistical stratified sentence split preserving non-O label distribution (with best-effort train coverage) | 5 | 0.743834907 | 0.025818097 |
| before_exp01_baseline | Regular NER with DictaBERT | 5 | 0.724011315 | 0.027596507 |

## Improvement Over Baseline

This table shows how much each strategy improved F1 compared with the random baseline.

| Variant | Delta Description | Seeds | Delta F1 |
|---|---|---:|---:|
| delta_after_inverse_freq_log_scaled_minus_baseline | Delta: Inv-freq log-scaled - baseline | 5 | **+0.154236704** |
| delta_after_inverse_freq_weighted_minus_baseline | Delta: Inverse-freq weighted - baseline | 5 | +0.148137525 |
| delta_after_inverse_freq_token_weighted_minus_baseline | Delta: Inv-freq token-weighted - baseline | 5 | +0.139702157 |
| delta_after_inverse_freq_eval_guaranteed_minus_baseline | Delta: Inv-freq eval-guaranteed - baseline | 5 | +0.095531475 |
| delta_after_rare_boosted_minus_baseline | Delta: Rare-label boosted - baseline | 5 | +0.083808891 |
| delta_after_minmax_equalized_minus_baseline | Delta: Min-max equalized - baseline | 5 | +0.037196875 |
| delta_after_label_aware_split_minus_baseline | Delta: Label-aware greedy - baseline | 5 | +0.019823592 |

## Plain-Language Interpretation

The results are very clear.

### 1. Random splitting was the weakest option

The baseline achieved **0.7240 F1**.

That means a simple random split leaves useful performance on the table.

### 2. Every label-aware method helped

All non-baseline strategies improved over the random split.

Even the weakest improvement among the smarter methods, **label-aware greedy**, still improved F1 by about **0.0198**.

### 3. Rare-label-aware methods helped the most

The strongest strategies were all based on giving more importance to rare labels:

- **after_inverse_freq_log_scaled**
- **after_inverse_freq_weighted**
- **after_inverse_freq_token_weighted**

These were the top three methods.

### 4. The best overall strategy was log-scaled inverse-frequency

The top result in this folder is:

- **after_inverse_freq_log_scaled**
- Mean F1: **0.878248018**
- Improvement over baseline: **+0.154236704**
- Standard deviation: **0.010825350**

This is important because it was not only the best-performing method, but also one of the most stable methods across seeds.

In simple terms, it achieved the best balance between:

- prioritizing rare labels,
- avoiding overly aggressive weighting,
- and producing consistent results.

## What The Files Mean

### Main result files

- `sentence_split_strategy_latest.xlsx`
  - Main Excel workbook for the most recent run.
- `latest.json`
  - Latest machine-readable summary.

### CSV summaries

- `sentence_split_strategy_metric_stats_latest.csv`
  - Mean/std statistics for each variant.
- `sentence_split_strategy_per_seed_latest.csv`
  - Raw results for each seed and each variant.
- `sentence_split_strategy_thesis_summary_latest.csv`
  - Thesis-style summary table.
- `sentence_split_strategy_training_label_count_latest.csv`
  - Before/after label-frequency analysis.

### Split files

- `splits/`
  - Saved train/evaluation sentence sets.
  - These are reused by downstream experiments so the exact same split policy can be tested in other NER architectures.

## Recommended Reading Order

If you want to understand this folder quickly, read the files in this order:

1. `sentence_split_strategy_latest.xlsx`
2. `sentence_split_strategy_metric_stats_latest.csv`
3. `sentence_split_strategy_training_label_count_latest.csv`
4. `splits/split_meta.json`

## Bottom Line

The main takeaway from this folder is:

> Better sentence splitting alone can produce a large improvement in NER performance.

The strongest method in this study was **log-scaled inverse-frequency splitting**, which raised mean F1 from **0.7240** to **0.8782**.

That makes this folder important not just as a record of one experiment, but as evidence that dataset partitioning is a major methodological choice in the thesis.