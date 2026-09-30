# Experiment 07 â€” Sentence Split Strategy Comparison

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

So Experiment 07 compares **3 different sentence-level splitting methods**.
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

In BIO tagging, the label `O` means â€œnot an entityâ€.

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

> The training set should â€œlook likeâ€ the full dataset as much as possible.

For example, if the full dataset contains:

- 40% person labels
- 35% location labels
- 20% organization labels
- 5% rare labels

then a good training split should have similar proportions.

---

## The 3 Methods â€” Simple Explanations

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

### Why it is called â€œgreedyâ€

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

## 3. Multilabel Stratified (Paper-Style Tie-Breaking)

### Main idea

This method treats each sentence as a **multi-label example** and tries to keep
label proportions similar across train and eval, using a paper-style iterative
stratification procedure (Sechidis et al., 2011 inspired).

### How it works

1. Represent each sentence by the set of non-`O` labels it contains.
2. Compute how many examples of each label should ideally go to train and eval.
3. Pick the rarest remaining label first.
4. For each sentence containing that label, choose the target fold using a more
   explicit tie-breaking order:
   - first by the remaining need for that label,
   - then by the remaining total capacity of the fold,
   - then randomly if still tied.
5. Update label targets and fold capacities after each assignment.
6. Finish unresolved sentences and rebalance to exact split size.

### Why this matters

This variant is the **only** multilabel-stratified train/eval split in the
pipeline: tie-breaking follows rare-label need, then fold capacity, then random.

### In one sentence

**This method preserves per-label proportions in train and eval using paper-style iterative stratification tie-breaking.**

---

## Comparing the Methods in One Table

| Method | Main Goal | Main Strength | Main Weakness |
|--------|-----------|---------------|---------------|
| Baseline (simple random) | Easy random split | Very simple | Ignores label balance |
| Label-aware greedy | Match train to full dataset | Good overall balance | May not emphasize very rare labels enough |
| Multilabel stratified (paper-style) | Preserve proportional label distribution with paper-style tie-breaking | Closer to the original iterative stratification procedure | More complex; heuristic on co-occurring labels |

---

## A Good Way to Teach This in a Beginner Class

You can explain the 3 methods in two teaching groups.

### Group A â€” The Simple Starting Point

- **Method 1: Baseline random**

This is what beginners already know: shuffle and split.

### Group B â€” Methods That Try to Make Training Better

- **Method 2: Label-aware greedy**
- **Method 3: Multilabel stratified (paper-style)**

These methods try to improve training quality by choosing better sentences.

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

Here is a short classroom summary of all 3 methods:

1. **Baseline random**: just shuffle and split.
2. **Label-aware greedy**: choose sentences that make training look like the full dataset.
3. **Multilabel stratified (paper-style)**: distribute each label proportionally across train and evaluation with paper-style tie-breaking.

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
