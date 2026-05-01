# Experiment 07 — Sentence Split Strategy Comparison

## What This Experiment Does

This experiment studies a very simple but very important question:

**How should we split the dataset into training sentences and evaluation sentences?**

In many machine-learning projects, people split data randomly and move on.
That is easy, but in NER (Named Entity Recognition) it can create a serious
problem:

- some labels are **common**
- some labels are **rare**
- after a random split, the training set may contain too few examples of the rare labels
- the evaluation set may become easier or harder just because of the split

So Experiment 07 compares **9 different sentence-level splitting methods**.
Each method creates a 70% training set and a 30% evaluation set, but each one
chooses the sentences in a different way.

The goal is to answer:

1. Which split method gives the best F1 score?
2. Which split method gives the fairest label distribution?
3. Which split method is most reliable for later experiments?

---

## First, a Very Simple Intuition

Imagine you have 10 sentences:

- 6 sentences contain only common labels
- 3 sentences contain medium-frequency labels
- 1 sentence contains a very rare label

If you split randomly, that rare-label sentence may:

- land in training
- or land in evaluation
- or be the only example in one side of the split

That means your result may depend more on luck than on method.

Experiment 07 tries to reduce that luck.

---

## Important Basic Ideas

### 1. Sentence-Level Split

The split is done **by sentence**, not by token.

That means a whole sentence is assigned either to:

- **training**, or
- **evaluation**

This avoids data leakage. We never want half of a sentence in train and the
other half in eval.

### 2. Non-O Labels Matter Most

In BIO tagging, the label `O` means “not an entity”.

Example:

- `O` = ordinary word
- `B-PER` = beginning of a person name
- `I-PER` = continuation of a person name
- `B-LOC` = beginning of a location

Most tokens are usually `O`, so if we only look at total label counts, the data
can look balanced even when the important entity labels are badly imbalanced.

That is why most methods in this experiment focus on the distribution of
**non-O labels**.

### 3. Train Distribution vs Full-Dataset Distribution

A central idea in this experiment is:

> The training set should “look like” the full dataset as much as possible.

For example, if the full dataset contains:

- 40% person labels
- 35% location labels
- 20% organization labels
- 5% rare labels

then a good training split should have similar proportions.

---

## The 9 Methods — Simple Explanations

## 1. Baseline (Simple Random)

### Main idea

This is the simplest method.
We shuffle the sentence list randomly and take:

- first 70% for training
- last 30% for evaluation

### How it works

1. Put all sentences in one list.
2. Shuffle them using a random seed.
3. Cut the list at 70%.
4. Use the first part as training and the second part as evaluation.

### Why people use it

Because it is easy, fast, and standard.

### Problem

It does **not** check whether rare labels are distributed fairly.
A rare label may appear too little in training or too much in evaluation.

### Beginner example

Suppose only 3 sentences contain `B-MISC`.
After random split:

- maybe all 3 go to eval
- maybe only 1 goes to train
- maybe none go to eval

So performance can change just because of chance.

### In one sentence

**Random split is simple, but it ignores label balance.**

---

## 2. Label-Aware Greedy

### Main idea

This method tries to build the training set so that the distribution of
**non-O labels** in training is close to the distribution in the full dataset.

### How it works

1. Count the label frequencies in the full dataset.
2. Start with an empty training set.
3. Add sentences one by one.
4. Each time, choose a sentence that makes the training-label distribution
   closer to the full-dataset distribution.
5. Stop when the training set reaches 70% of the sentences.

### Why it is called “greedy”

Because at each step it makes the **best local choice available at that moment**.
It does not test every possible combination of sentences.

### Why it helps

It reduces the chance that training becomes dominated by only common labels.

### Beginner example

Suppose training currently has too few `B-LOC` labels.
If a candidate sentence contains several `B-LOC` tokens, that sentence becomes
more attractive and is more likely to be added.

### In one sentence

**This method actively tries to keep the training label distribution similar to the whole dataset.**

---

## 3. Rare-Label Boosted

### Main idea

This method gives extra importance to sentences that contain **rare labels**.

### How it works

1. Compute how common or rare each label is.
2. Give higher score to rare labels.
3. Score each sentence based on the rare labels it contains.
4. Prefer high-scoring sentences for training.

### Why it helps

Rare labels are exactly the labels that models usually fail on.
If we make sure training contains more of them, the model gets a better chance
to learn them.

### Beginner example

Sentence A contains:
- `B-PER`, `I-PER` (common)

Sentence B contains:
- `B-CER`, `I-CER` (rare)

This method will prefer Sentence B more than a normal random or balanced split,
because it contains information the model is less likely to see elsewhere.

### Risk

If we push too hard toward rare labels, the training set may become less similar
to the real overall dataset.

### In one sentence

**This method intentionally favors sentences containing rare entity types.**

---

## 4. Inverse-Frequency Weighted

### Main idea

Instead of simply saying “rare labels matter more”, this method gives each label
a weight based on its frequency:

- common label -> small weight
- rare label -> large weight

Usually the idea is:

$$
weight(label) \propto \frac{1}{frequency(label)}
$$

### How it works

1. Count how often each label appears.
2. Convert each label count into a weight.
3. Score a sentence by adding up the weights of the labels inside it.
4. Prefer higher-scoring sentences for training.

### Why it helps

This creates a smoother version of “rare-label boosted”.
Instead of only saying “rare vs not rare”, it uses a continuous scale.

### Beginner example

Suppose:

- `B-PER` appears 1000 times
- `B-LOC` appears 300 times
- `B-MISC` appears 20 times

Then `B-MISC` gets much larger weight than `B-PER`.
A sentence containing `B-MISC` becomes much more valuable for the training set.

### In one sentence

**This method scores sentences using mathematical weights that reward rare labels more strongly than common ones.**

---

## 5. Min-Max Equalized

### Main idea

This method tries to reduce the biggest differences between:

- label frequencies in the training split
- label frequencies in the full dataset

It aims for a more even balance across labels.

### How it works

1. Measure the full-dataset frequency of each label.
2. Measure the current training frequency of each label.
3. Look at the gaps.
4. Prefer sentences that reduce the largest gaps.

### Why it helps

Sometimes one or two labels are much more underrepresented than the others.
This method focuses on correcting those strong imbalances.

### Beginner example

Suppose training already matches the full dataset for:

- person
- location
- organization

but is still missing many `B-EVE` labels.
Then this method prefers sentences containing `B-EVE` to reduce that worst gap.

### In one sentence

**This method tries to “flatten” the biggest imbalances between the training set and the full dataset.**

---

## 6. Inverse-Frequency Token-Weighted

### Main idea

This is similar to Method 4, but it scores at the **token level** more directly.

That means if a sentence contains many rare tokens, it receives more total score.

### Difference from Method 4

- Method 4 thinks more at the sentence label level
- Method 6 gives stronger emphasis to the **number of rare tokens** inside the sentence

### How it works

1. Count label frequencies.
2. Give inverse-frequency weight to each label.
3. For each token in a sentence, add the weight of its label.
4. Sentences with more rare-label tokens get higher score.

### Beginner example

Sentence A:
- one rare token

Sentence B:
- four rare tokens

Method 6 will often prefer Sentence B more strongly than Method 4, because it
rewards the total rare-token mass inside the sentence.

### In one sentence

**This method is like inverse-frequency weighting, but it pays more attention to how many rare tokens a sentence contains.**

---

## 7. Inverse-Frequency Eval-Guaranteed

### Main idea

This method builds on inverse-frequency weighting, but adds an extra rule:

> every label should appear at least once in the evaluation split, if possible.

### Why this matters

If a label does not appear in evaluation, then we cannot really judge whether the
model can recognise it.

Example:

- if `B-CER` appears only in training and never in eval,
  then the evaluation score says nothing about that label

### How it works

1. Use inverse-frequency weighting to choose good training sentences.
2. While splitting, make sure evaluation still keeps at least one example of
   each label whenever possible.
3. Avoid putting every rare-label sentence into training.

### Beginner example

Suppose there are only 2 sentences with a rare label.
A normal rare-label method might place both in training.
This method tries to keep at least one of them in evaluation.

### Trade-off

This helps evaluation fairness, but it can slightly reduce how many rare labels
training gets.

### In one sentence

**This method balances rare-label training value with the need to still test those labels in evaluation.**

---

## 8. Inverse-Frequency Log-Scaled

### Main idea

This method is like inverse-frequency weighting, but it uses a **logarithm** to
soften extreme weights.

Why is that useful?
Because very rare labels can otherwise get huge weights and dominate the split.

### Simple intuition

Without log scaling:

- very common label -> tiny score
- ultra-rare label -> enormous score

With log scaling:

- very common label -> still small score
- ultra-rare label -> important, but not absurdly dominant

A typical idea is:

$$
score(label) = \log\left(1 + \frac{max\_count}{count(label)}\right)
$$

### How it works

1. Count label frequencies.
2. Compute inverse-frequency style rarity.
3. Apply log scaling to reduce extreme values.
4. Score sentences using those softer rarity scores.

### Why it helps

This often gives a better compromise:

- rare labels are still rewarded
- one extremely rare label does not control everything

### Beginner example

Suppose:

- `B-PER` appears 1000 times
- `B-MISC` appears 2 times

Plain inverse weighting may push almost every choice toward `B-MISC`.
Log scaling still values `B-MISC` highly, but keeps the method more stable.

### In one sentence

**This method rewards rarity, but in a more controlled and stable way.**

---

## 9. Multilabel Stratified (Iterative Stratification)

### Main idea

This method treats each sentence as a **multi-label example** and tries to keep
the proportion of each label similar in both training and evaluation.

It is closer to classical multilabel stratification methods than the earlier
heuristics.

### How it works

1. Represent each sentence by the set of non-`O` labels it contains.
2. Compute how many examples of each label should go to train (70%) and eval
   (30%).
3. Process labels from rarest to most common.
4. For each sentence containing the current label, assign it to the side
   (train/eval) that still needs those labels more.
5. Update counts and continue until all labeled sentences are assigned.
6. Fill any remaining unlabeled or unresolved sentences to meet exact split
   size.

### Why it helps

It systematically reduces label-distribution drift between train and eval,
especially for labels that co-occur with others.

### Beginner example

Suppose `B-CER` and `B-EVE` often appear together in a small number of
sentences. A simple rare-label method may over-concentrate those sentences in
train. Multilabel stratification tries to distribute those co-occurrences more
proportionally across both sides.

### In one sentence

**This method uses iterative multilabel stratification to preserve per-label proportions in both training and evaluation.**

---

## Comparing the Methods in One Table

| Method | Main Goal | Main Strength | Main Weakness |
|--------|-----------|---------------|---------------|
| Baseline (simple random) | Easy random split | Very simple | Ignores label balance |
| Label-aware greedy | Match train to full dataset | Good overall balance | May not emphasize very rare labels enough |
| Rare-label boosted | Prioritize rare labels | Better rare-label exposure | Can distort overall distribution |
| Inverse-freq weighted | Reward rare labels mathematically | Flexible and principled | Can overweight extreme rarity |
| Min-max equalized | Reduce biggest distribution gaps | Good at fixing strong imbalance | May be less intuitive |
| Inv-freq token-weighted | Reward sentences with many rare tokens | Strong rare-token coverage | Can prefer dense rare-token sentences too much |
| Inv-freq eval-guaranteed | Preserve rare labels in eval too | Better fairness in testing | Training may get fewer rare examples |
| Inv-freq log-scaled | Reward rarity but gently | Stable and robust | Slightly less aggressive on ultra-rare labels |
| Multilabel stratified | Preserve proportional label distribution in both folds | Principled and balanced for multi-label co-occurrence | More complex and less intuitive to implement |

---

## A Good Way to Teach This in a Beginner Class

You can explain the 9 methods in three teaching groups.

### Group A — The Simple Starting Point

- **Method 1: Baseline random**

This is what beginners already know: shuffle and split.

### Group B — Methods That Try to Make Training Better

- **Method 2: Label-aware greedy**
- **Method 3: Rare-label boosted**
- **Method 4: Inverse-freq weighted**
- **Method 5: Min-max equalized**
- **Method 6: Inv-freq token-weighted**
- **Method 8: Inv-freq log-scaled**
- **Method 9: Multilabel stratified**

These methods all try to improve training quality by choosing better sentences.

### Group C — Method That Also Protects Evaluation

- **Method 7: Inv-freq eval-guaranteed**

This one teaches an important scientific idea:

> it is not enough to train well; we must also evaluate fairly.

---

## What Students Should Remember

If you teach this experiment to beginners, the most important lesson is:

> **Data splitting is not just a technical detail. It can strongly change model performance.**

Students should remember these core messages:

1. **Random split is simple, but not always fair.**
2. **Rare labels need special care.**
3. **A good training set should resemble the real dataset.**
4. **Evaluation must also remain meaningful.**
5. **The best method is often a balance between rarity and stability.**

---

## Short Teaching Summary

Here is a short classroom summary of all 9 methods:

1. **Baseline random**: just shuffle and split.
2. **Label-aware greedy**: choose sentences that make training look like the full dataset.
3. **Rare-label boosted**: push rare-label sentences into training.
4. **Inverse-freq weighted**: give rare labels bigger mathematical weight.
5. **Min-max equalized**: reduce the biggest label-balance gaps.
6. **Inv-freq token-weighted**: reward sentences with many rare-label tokens.
7. **Inv-freq eval-guaranteed**: keep rare labels visible in evaluation too.
8. **Inv-freq log-scaled**: reward rare labels, but not too aggressively.
9. **Multilabel stratified**: distribute each label proportionally across train and evaluation.

---

## Final Takeaway

Experiment 07 shows that **how we split the data can matter almost as much as
which model we train**.

A random split is easy, but smarter split strategies can:

- improve label coverage
- make training fairer
- make evaluation more meaningful
- improve downstream F1 scores

That is why Experiment 07 is important for the rest of the thesis: it finds the
most useful sentence-splitting strategy and saves those splits for reuse in later
experiments.
