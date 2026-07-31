# Theisis Overview: Formal Methodology of the Cross-Data Multi-Model NER Pipeline

## 1. Purpose of This Methodology

This document explains the scientific architecture triggered by the command:

```bash
python run_cross_data_model_comparison.py \
--resume \
--base-mode auto \
--experiments 01,04,05_ready,06_ready,06_svm_ready \
--models dictabert,berel,hero,alephbertgimmel \
--condition-sources exp07,exp07+aug \
--num-seeds 20
```

The command runs a controlled comparison of Hebrew Named Entity Recognition (NER) methods. It is not only a script that calls other scripts. It is a complete experimental architecture with:

1. several Hebrew transformer encoders,
2. several train/evaluation split strategies,
3. augmentation-based data variants,
4. base NER systems,
5. post-processing systems,
6. fusion systems,
7. paired multi-seed statistical testing,
8. cached base artifacts for reproducibility and efficiency.

The main scientific question is:

> How do different Hebrew transformer models behave under different sentence split and augmentation conditions, and can cascaded or fused prediction architectures improve entity-level NER performance?

The pipeline compares five **core** experiments (01–06 ready track):

| Experiment ID | Method name | Core idea |
|---|---|---|
| `01` | Regular NER | Single transformer token-classification model predicts full BIO entity labels directly. |
| `04` | AUC Cascaded Pipeline | NER is decomposed into entity detection, BIO position, and entity type prediction. |
| `05_ready` | Cascaded B/I Consistency | Post-processes `04` outputs to repair inconsistent `B-X` followed by `I-Y` predictions. |
| `06_ready` | Confidence Fusion | Combines `01` and `04`; if they disagree, the more confident source wins. |
| `06_svm_ready` | SVM Router Fusion | Combines `01` and `04`; an SVM learns which source to trust on disagreement tokens. |

**Optional extension — Experiment 10 (BERT-CRF track):** same cross-comparison runner, separate base cache, adds CRF decoding and CRF fusion (see **Section 12A** and `experiments/experiment_10_README.md`).

| Experiment ID | Method name | Core idea |
|---|---|---|
| `10_regular` | Regular BERT-CRF | Exp01-style single pass, but emissions + **linear-chain CRF** (Viterbi decode); **O-bias = 6**. |
| `10_cascade` | Cascaded + CRF | Exp04-style three heads **plus** full-tag CRF head; Step-3 **B/I consistency** after decode. |
| `10_fusion_ready` | CRF confidence fusion | Fuses `10_regular` and `10_cascade` ready Excel outputs (no retraining). |
| `10_svm_ready` | CRF SVM router | Same router idea as `06_svm_ready` on CRF sources. |

The selected models are:

| Model key | Model name | Scientific role |
|---|---|---|
| `dictabert` | DictaBERT | General-purpose Hebrew BERT baseline. |
| `berel` | BEREL 3.0 | Hebrew model with Biblical/Rabbinical orientation. |
| `hero` | HeRo | Hebrew RoBERTa-style model. |
| `alephbertgimmel` | AlephBERT-Gimmel | Hebrew BERT-family model. |

The selected condition sources are:

| Source | Meaning |
|---|---|
| `exp07` | Sentence split strategies without augmentation. |
| `exp07+aug` | The same Exp07 split variants, but the training side is augmented using the Exp08 LLM mask-fill augmentation mechanism. |

The command uses `--num-seeds 20`, so the design is repeated with seeds:

$$
S = \{42,43,44,\ldots,61\}.
$$

This makes the experiment a paired repeated-measures design rather than a one-off model run.

---

## 2. Beginner-Friendly View of the Whole Pipeline

Think of the pipeline as a large table of jobs. Each job has four coordinates:

$$
\text{job} = (m, e, c, s),
$$

where:

- $m$ is the model,
- $e$ is the experiment/method,
- $c$ is the data condition,
- $s$ is the random seed.

For this command:

$$
M = \{\text{DictaBERT}, \text{BEREL}, \text{HeRo}, \text{AlephBERT-Gimmel}\},
$$

$$
E = \{01,04,05\_ready,06\_ready,06\_svm\_ready\},
$$

$$
S = \{42,\ldots,61\}.
$$

The data condition set $C$ is built from saved Exp07 split variants and Exp07+Aug variants. In the current architecture, one excluded split variant is intentionally removed from the cross-comparison:

$$
\text{Excluded} = \{\text{after\_multilabel\_stratified}\}.
$$

So the practical number of data conditions depends on the saved Exp07 metadata, after exclusions. If three Exp07 variants are available and three matching Exp07+Aug variants are available, then:

$$
|C| = 3 + 3 = 6.
$$

The number of seeded conditions is:

$$
|C_S| = |C| \times |S|.
$$

If $|C|=6$ and $|S|=20$:

$$
|C_S| = 6 \times 20 = 120.
$$

The maximum number of result rows is:

$$
|M| \times |E| \times |C_S| = 4 \times 5 \times 120 = 2400.
$$

A beginner can understand the command as:

1. prepare split and augmentation files,
2. for every model,
3. for every experiment,
4. for every data condition,
5. for every seed,
6. run or reuse the method,
7. record F1, precision, recall,
8. aggregate results,
9. test whether differences are statistically meaningful.

---

## 3. Data Representation

The corpus is a Hebrew NER dataset stored as token-level BIO labels. Each token has one label:

$$
y_i \in \mathcal{Y},
$$

where:

$$
\mathcal{Y} = \{O\} \cup \{B\text{-}t, I\text{-}t : t \in \mathcal{T}\}.
$$

Here:

- $O$ means the token is not part of an entity,
- $B\text{-}t$ means the token begins an entity of type $t$,
- $I\text{-}t$ means the token continues an entity of type $t$,
- $\mathcal{T}$ is the set of entity types, such as `PER`, `LOC`, `ORG`, or other dataset-specific types.

A sentence is represented as:

$$
x^{(j)} = (w_1, w_2, \ldots, w_n),
$$

with labels:

$$
y^{(j)} = (y_1, y_2, \ldots, y_n).
$$

The saved split JSON format stores each sentence as:

```json
{
  "text": "token1 token2 token3",
  "labels": ["B-PER", "I-PER", "O"]
}
```

This matters because the pipeline splits at sentence level, not token level. Sentence-level splitting prevents leakage where part of a sentence appears in training and another part appears in evaluation.

---

## 4. Train/Evaluation Split Architecture

The command uses condition sources `exp07` and `exp07+aug`.

### 4.1 Exp07: Split Strategy Conditions

Exp07 creates different ways of dividing the same sentence set into training and evaluation partitions.

Let the full sentence set be:

$$
D = \{(x^{(j)}, y^{(j)})\}_{j=1}^{N}.
$$

Each split condition creates:

$$
D_{train}^{(c,s)} \subset D,
$$

$$
D_{eval}^{(c,s)} = D \setminus D_{train}^{(c,s)},
$$

with approximate ratio:

$$
\frac{|D_{train}^{(c,s)}|}{|D|} \approx 0.7,
\qquad
\frac{|D_{eval}^{(c,s)}|}{|D|} \approx 0.3.
$$

The split is sentence-level:

$$
D_{train}^{(c,s)} \cap D_{eval}^{(c,s)} = \varnothing.
$$

The split variants include ideas such as:

1. **Simple random split**: shuffle sentences and take the first 70% as training.
2. **Label-aware greedy split**: choose sentences so rare non-`O` labels are preserved in the training set.
3. **Paper-style iterative multilabel stratification**: treat each sentence as a set of labels and assign it to train/eval while preserving label proportions.

The cross-comparison runner excludes the older non-paper multilabel stratified variant, keeping the paper-relevant split variants.

### 4.2 Simple Random Split

The simplest split chooses a random permutation:

$$
\pi_s(D),
$$

where $s$ is the seed. Then:

$$
D_{train} = \pi_s(D)_{1:\lfloor 0.7N \rfloor},
$$

$$
D_{eval} = \pi_s(D)_{\lfloor 0.7N \rfloor + 1:N}.
$$

This is easy but risky. If a rare label appears in only a few sentences, all of those sentences may accidentally go to evaluation, leaving no training examples for that label.

### 4.3 Label-Aware Split

The label-aware split tries to preserve non-`O` entity label coverage.

For each label $\ell \neq O$, define the count in a sentence subset $A$:

$$
count_A(\ell) = \sum_{(x,y) \in A} \sum_{i=1}^{|x|} \mathbf{1}[y_i = \ell].
$$

The target training count is approximately:

$$
target_{train}(\ell) = 0.7 \cdot count_D(\ell).
$$

A natural split objective is to minimize distribution mismatch:

$$
\mathcal{L}_{split}(A)
=
\sum_{\ell \in \mathcal{Y}\setminus\{O\}}
\left(count_A(\ell) - target_{train}(\ell)\right)^2.
$$

The algorithm greedily builds a training set that keeps this loss small and tries to ensure that rare entity labels appear in the training set.

### 4.4 Multilabel Iterative Split

The paper-style iterative strategy treats each sentence as a multilabel item. A sentence may contain several entity labels, so it receives a set:

$$
L_j = \{\ell : \ell \neq O, \ell \text{ appears in sentence } j\}.
$$

The algorithm prioritizes rare labels first. For each label $\ell$, it estimates desired train/eval counts:

$$
desired_{train}(\ell) = 0.7 \cdot n_\ell,
$$

$$
desired_{eval}(\ell) = 0.3 \cdot n_\ell,
$$

where $n_\ell$ is the number of sentences containing label $\ell$.

It assigns sentences to the fold with the greater remaining need. For fold $f \in \{train, eval\}$, define:

$$
need_f(j) = \sum_{\ell \in L_j}
\left(desired_f(\ell) - current_f(\ell)\right).
$$

The chosen fold is:

$$
f^*(j) = \arg\max_f need_f(j),
$$

with deterministic seed-based tie-breaking.

---

## 5. Exp07+Aug: Augmented Split Conditions

The `exp07+aug` condition source starts with each Exp07 split variant and augments only the training set.

For a condition $c$ and seed $s$:

$$
D_{train}^{aug(c,s)} = D_{train}^{(c,s)} \cup G^{(c,s)},
$$

where $G^{(c,s)}$ is a generated set of synthetic training sentences.

The evaluation set is not augmented:

$$
D_{eval}^{aug(c,s)} = D_{eval}^{(c,s)}.
$$

This is a critical constraint. Augmenting evaluation data would change the test target and make results unfair. The method only increases the training examples.

### 5.1 Rare-Label Motivation

For each entity label $\ell$, define sentence frequency:

$$
f(\ell) = \sum_{j=1}^{N} \mathbf{1}[\ell \in L_j].
$$

Rare labels have smaller $f(\ell)$. The augmentation method tries to reduce imbalance by generating more examples for underrepresented labels.

Let:

$$
f_{max} = \max_{\ell} f(\ell).
$$

A simple deficit score is:

$$
\Delta(\ell) = f_{max} - f(\ell).
$$

The generation multiplier $r$ is controlled by `THESIS_EXP08_MULTIPLIER`, defaulting to $3$. The approximate target number of generated examples for label $\ell$ is:

$$
g(\ell) \approx r \cdot \Delta(\ell).
$$

### 5.2 Mask-Fill Generation

The augmentation mechanism uses a masked-language-model style approach. For a sentence containing an entity token, the method masks a relevant position and asks the language model to propose replacements.

A sentence:

$$
x = (w_1,\ldots,w_k,\ldots,w_n)
$$

is transformed into:

$$
x_{mask} = (w_1,\ldots,[MASK],\ldots,w_n).
$$

The language model estimates:

$$
P(w \mid x_{mask}).
$$

Candidate replacements are selected from entity-compatible vocabulary items. The generated sentence keeps the context but varies the entity surface form.

The final augmented training set is:

$$
D_{train}^{aug} = D_{train} \cup \{\tilde{x}_1,\tilde{x}_2,\ldots,\tilde{x}_K\}.
$$

The labels for generated sentences preserve the intended BIO structure. The purpose is not to invent a new evaluation target, but to expose the model to more rare-label contexts during training.

---

## 6. Model Resolution and Reproducibility Controls

For every selected model key, the runner maps it to a model identifier:

$$
m \mapsto \text{HuggingFace model ID or local model path}.
$$

The runner prefers local model files when available. If local files are found, it sets offline flags so the transformer library does not unnecessarily download model files.

For each run, the following environment variables are used to bind the scientific context:

| Variable | Meaning |
|---|---|
| `THESIS_MODEL_NAME` | Which transformer model to use. |
| `THESIS_SPLIT_SEED` | Which random seed controls the current condition. |
| `THESIS_PRESPLIT_TRAIN_JSON` | Exact train split file. |
| `THESIS_PRESPLIT_EVAL_JSON` | Exact evaluation split file. |
| `THESIS_CURRENT_EXP_ID` | Current experiment ID for checkpoint/model saving. |
| `THESIS_CURRENT_CONDITION_KEY` | Current condition key for artifact isolation. |
| `THESIS_READY_EXP01_XLSX` | Exp01 output file used by ready experiments. |
| `THESIS_READY_EXP04_XLSX` | Exp04 output file used by ready experiments. |

The important beginner idea is:

> Ready experiments do not guess which previous output to use. The runner explicitly points them to the matching Exp01 and Exp04 files for the same model, condition, and seed.

---

## 7. Artifact Reuse and `--base-mode auto`

Experiments `01` and `04` are expensive because they train transformer-based systems. Experiments `05_ready`, `06_ready`, and `06_svm_ready` are cheaper because they operate on already saved predictions.

Experiment **10** adds another **training** pair (`10_regular`, `10_cascade`) with the same cost profile as `01`/`04`, plus **inference-only** fusion IDs (`10_fusion_ready`, `10_svm_ready`) analogous to the 06 ready track. Base artifacts are stored in `cross_comparison_base_crf_ready_index.json` (separate from the Exp01/Exp04 cache).

The command uses:

```text
--base-mode auto
```

This means:

1. if valid Exp01 and Exp04 artifacts already exist for the same model and condition, reuse them;
2. otherwise, train Exp01 and Exp04 once;
3. save their output paths in the base artifact index;
4. pass those output paths into ready experiments.

The cache key is conceptually:

$$
k = (m, c, path(D_{train}), path(D_{eval})).
$$

An artifact is valid only if all required files exist:

$$
\text{valid}(k) =
\mathbf{1}[F_{01}^{xlsx} \land F_{01}^{json} \land F_{04}^{xlsx} \land F_{04}^{json}].
$$

The `--resume` flag adds another layer. If a previous cross-comparison checkpoint already contains a successful row for a given:

$$
(m,e,c,s),
$$

that row is skipped. Failed rows are not treated as complete and may be retried.

---

## 8. Experiment 01: Regular Transformer NER

Experiment `01` is the baseline direct NER architecture.

### 8.1 Input and Output

Input sentence:

$$
x = (w_1,w_2,\ldots,w_n).
$$

Gold labels:

$$
y = (y_1,y_2,\ldots,y_n), \qquad y_i \in \mathcal{Y}.
$$

The transformer tokenizer may split words into subword pieces. The model predicts labels at token/subtoken level, then outputs token-level predictions aligned back to the dataset.

### 8.2 Transformer Encoder

Each token is converted into a contextual vector:

$$
H = Transformer_m(x),
$$

where:

$$
H = (h_1,h_2,\ldots,h_n), \qquad h_i \in \mathbb{R}^d.
$$

Here $m$ is one of the selected Hebrew transformer models.

### 8.3 Token Classification Head

A linear classification head maps each hidden vector to label logits:

$$
z_i = W h_i + b.
$$

The probability of label $k$ is computed with softmax:

$$
P(y_i=k \mid x) =
\frac{\exp(z_{i,k})}{\sum_{k' \in \mathcal{Y}} \exp(z_{i,k'})}.
$$

The predicted label is:

$$
\hat{y}_i = \arg\max_{k \in \mathcal{Y}} P(y_i=k \mid x).
$$

### 8.4 Training Objective

The model is trained by minimizing cross-entropy over valid tokens:

$$
\mathcal{L}_{NER}
=
-\sum_{i=1}^{n}
\log P(y_i \mid x).
$$

In implementation, special tokens and ignored subtokens receive label `-100`, so they do not contribute to the loss:

$$
\mathcal{L}_{NER}
=
-\sum_{i: y_i \neq -100}
\log P(y_i \mid x).
$$

### 8.5 Confidence Features Exported for Fusion

Exp01 exports more than the final label. For each token, it saves:

1. predicted label,
2. maximum probability,
3. entropy,
4. probability margin.

The maximum probability is:

$$
p_i^{reg} = \max_k P(y_i=k \mid x).
$$

Entropy is:

$$
H_i^{reg} = -\sum_{k \in \mathcal{Y}} P(y_i=k \mid x)\log(P(y_i=k \mid x)+\epsilon).
$$

The margin is the difference between the top two probabilities:

$$
margin_i^{reg} = p_{i,(1)} - p_{i,(2)}.
$$

A large margin means the model strongly prefers its top label over the second-best label.

---

## 9. Experiment 04: Cascaded Multi-Step NER

Experiment `04` decomposes NER into three simpler prediction problems.

Instead of directly predicting full labels like `B-PER`, the model predicts:

1. whether a token is an entity,
2. whether an entity token is `B` or `I`,
3. which entity type the token has.

This is called a cascaded architecture because the final prediction depends on several steps.

### 9.1 Label Decomposition

A full BIO label $y_i$ is decomposed into:

$$
e_i \in \{0,1\},
$$

$$
b_i \in \{0,1\},
$$

$$
t_i \in \mathcal{T}.
$$

Where:

- $e_i=1$ means token $i$ is part of an entity,
- $e_i=0$ means token $i$ is outside an entity,
- $b_i=1$ means `B`,
- $b_i=0$ means `I`,
- $t_i$ is the entity type.

For label `O`:

$$
e_i=0, \qquad b_i=-100, \qquad t_i=-100.
$$

The value `-100` means “ignore this token for that task.”

### 9.2 Shared Encoder with Three Heads

The cascaded model uses one shared transformer encoder:

$$
h_i = Encoder_m(x)_i.
$$

Then it applies three heads:

Entity detection head:

$$
z_i^{entity} = W_e h_i + b_e,
$$

BIO head:

$$
z_i^{bio} = W_b h_i + b_b,
$$

Type head:

$$
z_i^{type} = W_t h_i + b_t.
$$

The probabilities are:

$$
p_i^{entity} = \sigma(z_i^{entity}),
$$

$$
p_i^{bio} = \sigma(z_i^{bio}),
$$

$$
P(t_i=k \mid x) =
\frac{\exp(z_{i,k}^{type})}{\sum_{k' \in \mathcal{T}}\exp(z_{i,k'}^{type})}.
$$

Here $\sigma$ is the sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

### 9.3 Cascaded Decision Rule

The pipeline converts probabilities into decisions using thresholds.

Entity decision:

$$
\hat{e}_i =
\begin{cases}
1, & p_i^{entity} \geq \tau_e,\\
0, & p_i^{entity} < \tau_e.
\end{cases}
$$

BIO decision:

$$
\hat{b}_i =
\begin{cases}
B, & p_i^{bio} \geq \tau_b,\\
I, & p_i^{bio} < \tau_b.
\end{cases}
$$

Type decision:

$$
\hat{t}_i = \arg\max_{k \in \mathcal{T}} P(t_i=k \mid x).
$$

Final label:

$$
\hat{y}_i^{cas}=
\begin{cases}
O, & \hat{e}_i=0,\\
\hat{b}_i\text{-}\hat{t}_i, & \hat{e}_i=1.
\end{cases}
$$

### 9.4 Masked Conditional Training

The architecture trains later heads only where they make sense.

Entity loss is computed for all valid tokens:

$$
\mathcal{L}_{entity} = \sum_{i:e_i\neq -100} \ell_{bin}(z_i^{entity}, e_i).
$$

BIO loss is computed only for true entity tokens:

$$
\mathcal{L}_{bio} = \sum_{i:b_i\neq -100} \ell_{bin}(z_i^{bio}, b_i).
$$

Type loss is also computed only for true entity tokens:

$$
\mathcal{L}_{type} = \sum_{i:t_i\neq -100} \ell_{CE}(z_i^{type}, t_i).
$$

The total loss is weighted:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{entity}
+ \lambda_{bio}\mathcal{L}_{bio}
+ \lambda_{type}\mathcal{L}_{type}.
$$

The implementation uses:

$$
\lambda_{bio}=10,
\qquad
\lambda_{type}=5.
$$

### 9.5 Focal Loss for Imbalance

The binary heads can use focal loss. This is useful because most tokens are usually `O`, so easy negative examples can dominate training.

For a binary target $y \in \{0,1\}$ and predicted probability $p$, define:

$$
p_t =
\begin{cases}
p, & y=1,\\
1-p, & y=0.
\end{cases}
$$

Focal loss is:

$$
\mathcal{L}_{focal}
= -\alpha_t(1-p_t)^\gamma \log(p_t).
$$

With:

$$
\alpha=0.25,
\qquad
\gamma=2.0.
$$

Beginner explanation: if a token is already easy, then $p_t$ is high and $(1-p_t)^\gamma$ becomes small. The loss focuses more on hard examples.

### 9.6 Threshold Optimization

After training, the cascaded pipeline searches over threshold values:

$$
\tau_e, \tau_b \in \{0.10,0.15,0.20,\ldots,0.90\}.
$$

It chooses the pair that maximizes a combined score:

$$
(\tau_e^*,\tau_b^*)
=
\arg\max_{\tau_e,\tau_b}
\left(F1_{entity}(\tau_e)+F1_{bio}(\tau_b)\right).
$$

The final exported result uses the optimized thresholds.

### 9.7 BIO Constraint Enforcement

The cascaded system also repairs invalid BIO transitions. For example, an `I` tag should not start an entity immediately after `O`.

If:

$$
\hat{b}_i = I
\quad\text{and}\quad
(i=1 \text{ or } \hat{b}_{i-1}=O),
$$

then it changes:

$$
\hat{b}_i := B.
$$

This turns an invalid continuation into a valid beginning.

---

## 10. Experiment 05_ready: Cascaded B/I Consistency Post-Processing

Experiment `05_ready` does not train a new model. It loads Exp04 predictions and applies a local correction rule.

### 10.1 The Problem

A cascaded model can predict inconsistent neighboring labels, such as:

```text
B-PER I-LOC
```

This says: “start a person entity, then continue it as a location entity,” which is structurally inconsistent.

The problematic pattern is:

$$
\hat{b}_i = B,
\qquad
\hat{b}_{i+1} = I,
\qquad
\hat{t}_i \neq \hat{t}_{i+1}.
$$

### 10.2 Consistency Rule

Let $q_i$ be the BIO confidence for token $i$, saved as `bio_prob`.

If a `B-X` token is followed by `I-Y` and $X \neq Y$, the method trusts the token with higher BIO confidence.

Formally:

$$
(\hat{t}_i^*, \hat{t}_{i+1}^*) =
\begin{cases}
(\hat{t}_i, \hat{t}_i), & q_i \geq q_{i+1},\\
(\hat{t}_{i+1}, \hat{t}_{i+1}), & q_i < q_{i+1}.
\end{cases}
$$

Beginner explanation:

- if the `B` token is more confident, force the following `I` token to use the same entity type;
- if the `I` token is more confident, change the `B` token to match the `I` token type.

The method then reconstructs BIO labels and recomputes entity-level F1.

---

## 11. Experiment 06_ready: Confidence Fusion of Regular and Cascaded NER

Experiment `06_ready` combines two different prediction sources:

1. regular NER from Exp01,
2. cascaded NER from Exp04.

It does not retrain either source model.

### 11.1 Token Alignment

Exp01 and Exp04 outputs are inner-joined by:

$$
(sentence\_id, token\_idx).
$$

A token can be fused only if both systems produced a prediction for the same sentence and token index.

For token $i$:

- regular prediction: $\hat{y}_i^{reg}$,
- cascaded prediction: $\hat{y}_i^{cas}$,
- regular confidence: $p_i^{reg}$,
- cascaded confidence: $p_i^{cas}$.

### 11.2 Cascaded Confidence

Exp04 exports entity probability and BIO probability. The fusion loader converts these into a single confidence score.

If the cascaded model predicts `O`, confidence is:

$$
p_i^{cas} = 1 - p_i^{entity}.
$$

If it predicts an entity, confidence is:

$$
p_i^{cas} = p_i^{entity} \cdot p_i^{bio}.
$$

This means the cascaded system must be confident both that the token is an entity and that the BIO position is correct.

### 11.3 Fusion Rule

If both systems agree:

$$
\hat{y}_i^{fused} = \hat{y}_i^{reg} = \hat{y}_i^{cas}.
$$

If they disagree, choose the label from the more confident system:

$$
\hat{y}_i^{fused} =
\begin{cases}
\hat{y}_i^{reg}, & p_i^{reg} \geq p_i^{cas},\\
\hat{y}_i^{cas}, & p_i^{reg} < p_i^{cas}.
\end{cases}
$$

This is a simple but scientifically meaningful ensemble rule. It assumes that confidence is a useful proxy for correctness.

---

## 12. Experiment 06_svm_ready: SVM Router Fusion

Experiment `06_svm_ready` uses a learned router instead of a fixed confidence rule.

The central idea is:

> When Exp01 and Exp04 disagree, learn which source is more likely to be correct.

### 12.1 Disagreement Tokens

A disagreement token is one where:

$$
\hat{y}_i^{reg} \neq \hat{y}_i^{cas}.
$$

The SVM is trained only on disagreement tokens where exactly one source is correct.

Define:

$$
r_i = \mathbf{1}[\hat{y}_i^{reg} = y_i],
$$

$$
c_i = \mathbf{1}[\hat{y}_i^{cas} = y_i].
$$

The router target is:

$$
z_i =
\begin{cases}
regular, & r_i=1 \land c_i=0,\\
cascade, & r_i=0 \land c_i=1,\\
\text{discard}, & \text{otherwise}.
\end{cases}
$$

Ambiguous cases are discarded:

- both correct,
- both wrong.

### 12.2 Router Features

For each disagreement token, the feature vector includes numeric and categorical features.

Numeric features:

$$
\phi_i^{num} = [
 p_i^{reg},
 p_i^{cas},
 margin_i^{reg},
 margin_i^{cas},
 p_i^{reg}-p_i^{cas},
 |p_i^{reg}-p_i^{cas}|,
 \max(p_i^{reg},p_i^{cas})
].
$$

Categorical features:

$$
\phi_i^{cat} = [
BIO_i^{reg},
TYPE_i^{reg},
BIO_i^{cas},
TYPE_i^{cas}
].
$$

The full router feature vector is:

$$
\phi_i = [\phi_i^{num}, \phi_i^{cat}].
$$

Numeric features are standardized. Categorical features are one-hot encoded.

### 12.3 Linear SVM Objective

The router is a linear support vector classifier. In simplified binary form, it learns:

$$
f(\phi_i) = w^T\phi_i + b.
$$

The predicted source is:

$$
\hat{z}_i =
\begin{cases}
regular, & f(\phi_i) \geq 0,\\
cascade, & f(\phi_i) < 0.
\end{cases}
$$

The SVM minimizes hinge loss with regularization:

$$
\min_{w,b}
\frac{1}{2}\|w\|^2
+
C\sum_i \max(0, 1 - y_i^{svm}(w^T\phi_i+b)).
$$

The implementation uses:

$$
C=1.0.
$$

Class weights are balanced so that the router does not simply prefer the majority source.

### 12.4 Final SVM Fusion Rule

For agreement tokens:

$$
\hat{y}_i^{fused} = \hat{y}_i^{reg} = \hat{y}_i^{cas}.
$$

For disagreement tokens:

$$
\hat{y}_i^{fused} =
\begin{cases}
\hat{y}_i^{reg}, & \hat{z}_i = regular,\\
\hat{y}_i^{cas}, & \hat{z}_i = cascade.
\end{cases}
$$

If the router cannot be trained, for example because there are not enough usable disagreement examples, the method falls back to the confidence fusion rule.

Important limitation:

> The ready SVM variant trains and evaluates the router on the same ready output set. It is useful as an exploratory fusion architecture, but a stricter estimate would require a separate router train/test split.

---

## 12A. Experiment 10 — BERT-CRF Extension (Optional Cross-Comparison Branch)

Experiment 10 implements **future-work items** from the thesis plan: replace (or augment) independent softmax tagging with a **Conditional Random Field** so **BIO transition structure** is learned during training, not only repaired afterward (compare Exp05_ready).

It is **additive**: Experiments 01, 04, and 05/06 ready paths are unchanged. Exp10 uses its **own output folders** (`outputs/exp10_regular/`, `outputs/exp10_cascade/`, …) and **own base cache index**.

**Teaching documentation:** `experiments/experiment_10_README.md`  
**Code map:** `core/crf_layer.py`, `core/bert_crf_training.py`, `core/cascaded_crf_runtime.py`

Example command:

```bash
python run_cross_data_model_comparison.py \
  --experiments 10_regular,10_cascade,10_fusion_ready,10_svm_ready \
  --models dictabert,berel \
  --base-mode auto
```

### 12A.1 Experiment 10_regular — BERT + Emissions + CRF

Architecture:

1. Hebrew transformer encoder (same registry as Exp01).
2. Linear layer producing emission scores $e_t(k)$ for each tag $k$ at token $t$.
3. **Linear-chain CRF** with transition matrix $A$ of size $(K+2)\times(K+2)$ (START/STOP states).

Training minimizes **CRF negative log-likelihood** (forward algorithm for $\log Z(x)$). Inference uses **Viterbi** to obtain $\hat{\mathbf{y}}$.

**Class imbalance:** bias of the `O` tag in the emission layer is initialized to **6** (Souza et al. 2019).

Mathematically, for gold sequence $\mathbf{y}^*$:

$$
\mathcal{L}_{CRF} = \log Z(x) - s(\mathbf{y}^*),
\quad
s(\mathbf{y}) = \sum_t \left( A_{y_{t-1},y_t} + e_t(y_t) \right).
$$

This is the same structural idea as classical NER-CRF, but emissions come from **BERT** rather than hand-crafted features.

### 12A.2 Experiment 10_cascade — Cascaded Heads + Full-Tag CRF

Exp04 trains three heads (entity, B/I, type) and **composes** their outputs. Exp10 **retains** those heads for diagnostic step F1, and adds a **joint head** over full BIO-type labels with CRF loss.

**Pipeline span F1** for reporting uses **Viterbi** on the joint head (word-level aggregated emissions). Optional post-decode rule:

$$
\text{if } \hat{b}_i=B-X \text{ and } \hat{b}_{i+1}=I-Y \text{ with } X\neq Y,\ \text{reconcile types (Exp05 idea)}.
$$

Controlled by `THESIS_STEP3_BI_TYPE_RECONCILE=1` in the cascaded wrapper script.

### 12A.3 Experiment 10_fusion_ready — Confidence Fusion on CRF Outputs

Identical arbitration to Exp06_ready, but inputs are:

- regular CRF token predictions (`token_predictions` sheet), and
- cascaded CRF token predictions (`detailed_results`, `eval_mode=predicted`).

For token $i$, if $\hat{y}_i^{reg} \neq \hat{y}_i^{cas}$:

$$
\hat{y}_i^{fused} =
\begin{cases}
\hat{y}_i^{reg}, & p_i^{reg} \geq p_i^{cas},\\
\hat{y}_i^{cas}, & p_i^{reg} < p_i^{cas}.
\end{cases}
$$

### 12A.4 Experiment 10_svm_ready — SVM Router on CRF Disagreements

Same feature vector and LinearSVC objective as Section 12 (`06_svm_ready`), applied to **CRF** disagreement tokens. Router targets:

$$
z_i \in \{regular, cascade\}
$$

only when exactly one of $\hat{y}_i^{reg}, \hat{y}_i^{cas}$ equals $y_i$.

### 12A.5 Caching, Cleanup, and Error Analysis

**Base cache:** `_ensure_base_artifacts_crf` trains or reuses `10_regular` + `10_cascade` per $(m,c)$ and records paths in `cross_comparison_base_crf_ready_index.json`.

**Disk cleanup:** after training, checkpoint directories may be deleted (`THESIS_DELETE_MODELS_AFTER_TRAIN=1`) while **Excel/JSON metrics are retained**.

**Consolidated error analysis:** when any Exp10 ID is selected, the runner merges error-analysis workbooks into `outputs/cross_comparison/consolidated_error_analysis_exp10_<timestamp>.xlsx`.

---

## 13. Evaluation Metric: Entity-Level Precision, Recall, and F1

The primary metric is strict entity-level F1 using seqeval-style evaluation.

A predicted entity is correct only if both boundary and type match.

A true entity span is:

$$
(s,e,t),
$$

where:

- $s$ is start token index,
- $e$ is exclusive end token index,
- $t$ is entity type.

Let:

- $G$ be the set of gold spans,
- $P$ be the set of predicted spans.

Then:

$$
TP = |P \cap G|,
$$

$$
FP = |P \setminus G|,
$$

$$
FN = |G \setminus P|.
$$

Precision:

$$
Precision = \frac{TP}{TP+FP}.
$$

Recall:

$$
Recall = \frac{TP}{TP+FN}.
$$

F1:

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}.
$$

Beginner explanation:

- precision asks: “Of the entities the model predicted, how many were correct?”
- recall asks: “Of the true entities in the data, how many did the model find?”
- F1 balances both.

---

## 14. Paired Seed Design

The same seed list is used across all models, methods, and compatible conditions:

$$
S = \{42,43,\ldots,61\}.
$$

This creates paired observations. For example, the F1 of split-only seed 42 can be compared directly with split+augmentation seed 42.

For a pair of conditions $A$ and $B$, define:

$$
x_s = F1(A,s),
$$

$$
y_s = F1(B,s),
$$

$$
d_s = y_s - x_s.
$$

The mean difference is:

$$
\bar{d} = \frac{1}{n}\sum_{s=1}^{n}d_s.
$$

The sample standard deviation is:

$$
s_d = \sqrt{\frac{1}{n-1}\sum_{s=1}^{n}(d_s-\bar{d})^2}.
$$

The paired t-statistic is:

$$
t = \frac{\bar{d}}{s_d/\sqrt{n}}.
$$

The null hypothesis is:

$$
H_0: \mu_d = 0.
$$

The alternative hypothesis is:

$$
H_1: \mu_d \neq 0.
$$

The runner also computes a Wilcoxon signed-rank test, which is less dependent on normality assumptions.

For Wilcoxon:

1. compute differences $d_s$,
2. remove zero differences,
3. rank $|d_s|$,
4. attach signs,
5. test whether positive and negative signed ranks are balanced.

The significance threshold is:

$$
\alpha = 0.05.
$$

A result with:

$$
p < 0.05
$$

is marked as statistically significant.

---

## 15. Aggregation Across Runs

For each group:

$$
(m,e,c),
$$

the runner aggregates over seeds.

Mean F1:

$$
\overline{F1}_{m,e,c}
=
\frac{1}{|S|}\sum_{s \in S}F1_{m,e,c,s}.
$$

Standard deviation:

$$
SD(F1)_{m,e,c}
=
\sqrt{\frac{1}{|S|-1}\sum_{s\in S}(F1_{m,e,c,s}-\overline{F1}_{m,e,c})^2}.
$$

The same aggregation is also computed for precision and recall.

The runner computes deltas such as:

Exp07 split variant improvement:

$$
\Delta_{exp07}
=
\overline{F1}_{variant}-\overline{F1}_{baseline}.
$$

Exp07+Aug improvement over the matching split-only condition:

$$
\Delta_{aug}
=
\overline{F1}_{exp07+aug}-\overline{F1}_{exp07}.
$$

Positive delta means improvement.

Negative delta means degradation.

---

## 16. Data Flow of the Complete Command

The full command follows this order.

### Step 1: Parse Configuration

The runner reads:

- selected experiments,
- selected models,
- selected condition sources,
- seed count,
- resume mode,
- base artifact mode.

It validates that the experiment IDs and model keys exist.

### Step 2: Prepare Exp07 Splits

The runner checks:

```text
outputs/exp07/splits/split_meta.json
```

If the saved split metadata and files exist, it reuses them. If they are missing and `--exp07-source auto` is active, it regenerates Exp07 split artifacts.

### Step 3: Prepare Exp08 Machinery

Even though the selected condition sources are `exp07` and `exp07+aug`, the runner prepares Exp08 split/augmentation machinery because Exp07+Aug uses Exp08-style mask-fill augmentation.

### Step 4: Prepare Exp07+Aug Splits

For each Exp07 variant and each seed, the runner creates or reuses:

$$
D_{train}^{aug(c,s)},
\qquad
D_{eval}^{c,s}.
$$

The augmented files are saved under:

```text
outputs/exp07_augmented/splits/
```

### Step 5: Build Base Conditions

The runner creates a list of base condition dictionaries. Each condition stores:

- source,
- condition key,
- variant name,
- human-readable label,
- train split path,
- eval split path,
- baseline marker,
- optional seed-specific file map.

### Step 6: Expand Conditions by Seed

For every base condition $c$ and seed $s$:

$$
c_s = expand(c,s).
$$

The expanded condition receives a key like:

```text
exp07_after_label_aware_split__seed42
```

or:

```text
exp07aug_after_label_aware_split__seed42
```

### Step 7: Loop Over Models, Experiments, and Conditions

For every model $m$, experiment $e$, and seeded condition $c_s$, the runner executes one run.

The nested conceptual loop is:

$$
\text{for } m \in M:
\quad
\text{for } e \in E:
\quad
\text{for } c_s \in C_S:
\quad
run(m,e,c_s).
$$

### Step 8: Ensure Base Artifacts

For `01`, `04`, and all ready experiments **on the 01/04 track**, the runner ensures that matching Exp01 and Exp04 artifacts exist.

For Experiment **10**, Step 8 applies the same idea via `_ensure_base_artifacts_crf` for `10_regular` + `10_cascade` (separate index file).

If missing:

1. set train/eval split environment variables,
2. run Exp01,
3. run Exp04,
4. store artifact paths.

If already available:

1. reuse the existing files,
2. avoid retraining.

### Step 9: Run Ready Experiments

For `05_ready`, `06_ready`, and `06_svm_ready`, the runner injects:

```text
THESIS_READY_EXP01_XLSX
THESIS_READY_EXP04_XLSX
```

For `10_fusion_ready` and `10_svm_ready`, the runner injects:

```text
THESIS_READY_EXP10_REGULAR_XLSX
THESIS_READY_EXP10_CASCADE_XLSX
```

Then the ready experiment reads exactly those matched files.

### Step 10: Save Checkpoint After Every Run

After each run, the runner writes progress to:

```text
outputs/cross_comparison/cross_comparison_progress_latest.json
```

This supports `--resume`.

### Step 11: Aggregate Results and Export

At the end, the runner writes Excel and JSON outputs under:

```text
outputs/cross_comparison/
```

---

## 17. Output Files and Their Scientific Meaning

The main output workbook is:

```text
outputs/cross_comparison/cross_comparison_<timestamp>.xlsx
```

A latest copy is also written:

```text
outputs/cross_comparison/cross_comparison_latest.xlsx
```

The workbook contains sheets with different scientific roles.

| Sheet | Meaning |
|---|---|
| `summary_pivot` | One row per model and experiment, with mean F1 for each condition. |
| `all_runs` | Raw result table with every model/experiment/condition/seed row. |
| `deltas_exp07` | Difference between Exp07 split variants and Exp07 baseline. |
| `deltas_exp08` | Difference between Exp08 augmented and baseline if Exp08 conditions are included. |
| `deltas_exp07_aug` | Difference between Exp07+Aug and the matching Exp07 split-only condition. |
| `paired_tests` | Paired t-test and Wilcoxon test results across seeds. |
| `model_comparison` | Head-to-head model comparison if exactly two models are selected. |
| `variant_summary` | Aggregate statistics per condition across models and experiments. |
| `experiment_details` | File paths, status, timing, and detailed metadata. |
| `documentation` | Built-in explanation of workbook interpretation. |

The JSON output contains the same information in machine-readable format:

```text
outputs/cross_comparison/cross_comparison_<timestamp>.json
```

A latest copy is also written:

```text
outputs/cross_comparison/cross_comparison_latest.json
```

---

## 18. Methodological Constraints and Assumptions

### 18.1 Same Evaluation Set Within a Condition

For a given condition and seed, all experiments and models should use the same evaluation split:

$$
D_{eval}^{(c,s)}.
$$

This makes model and method comparisons fair.

### 18.2 No Evaluation Augmentation

For `exp07+aug`, only training data changes:

$$
D_{train} \rightarrow D_{train} \cup G.
$$

The evaluation set remains unchanged:

$$
D_{eval}^{aug} = D_{eval}.
$$

### 18.3 Ready Experiments Depend on Base Outputs

`05_ready`, `06_ready`, and `06_svm_ready` are not independent training methods. They are downstream methods applied to Exp01 and/or Exp04 outputs.

Experiment **10** ready fusion methods depend on **`10_regular` + `10_cascade`** outputs (not on Exp01/Exp04).

The dependency graph is:

```text
Exp01 ─┐
       ├── Exp06_ready
Exp04 ─┘

Exp04 ─── Exp05_ready

Exp01 ─┐
       ├── Exp06_svm_ready
Exp04 ─┘

Exp10_regular ─┐
               ├── Exp10_fusion_ready
Exp10_cascade ─┤
               └── Exp10_svm_ready
```

### 18.4 SVM Ready Fusion Is Exploratory

The SVM router uses labels from the ready evaluation outputs to learn which source is correct on disagreement tokens. Since it trains and evaluates on the same set, it should be interpreted as an upper-bound or exploratory router behavior, not a fully held-out router evaluation.

### 18.5 F1 Is Strict Entity-Level F1

A token-level partially correct prediction is not enough. The entity boundary and entity type must both match.

For example, if the true entity is:

```text
B-PER I-PER
```

but the prediction is:

```text
B-PER O
```

then the full entity span is not correct.

---

## 19. Scientific Architecture in One Formal Diagram

The command implements the following mathematical mapping:

$$
(D, M, C, S, E)
\longrightarrow
\{F1_{m,e,c,s}, Precision_{m,e,c,s}, Recall_{m,e,c,s}\}.
$$

Where:

$$
D = \text{Hebrew BIO-labeled NER corpus},
$$

$$
M = \text{selected Hebrew transformer models},
$$

$$
C = \text{split and augmentation conditions},
$$

$$
S = \text{paired seed set},
$$

$$
E = \text{selected NER architectures}.
$$

The result tensor is:

$$
R \in \mathbb{R}^{|M| \times |E| \times |C| \times |S| \times 3},
$$

where the final dimension stores:

$$
(F1, Precision, Recall).
$$

The analysis layer transforms $R$ into:

1. means,
2. standard deviations,
3. deltas,
4. paired statistical tests,
5. best-condition summaries,
6. model comparison tables.

---

## 20. Final Interpretation Guide

A reader should interpret the results as follows.

### If Exp01 performs best

The direct transformer token classifier is sufficient, and decomposition/fusion may not add value for that condition.

### If Exp04 performs best

The cascaded decomposition helps by separating the NER problem into easier subtasks:

$$
NER \approx EntityDetection + BIOPosition + EntityType.
$$

### If Exp05_ready improves over Exp04

Most improvement comes from repairing structurally inconsistent BIO/entity-type transitions.

### If Exp06_ready improves over both Exp01 and Exp04

Confidence-based fusion successfully exploits complementary strengths of the direct and cascaded systems.

### If Exp06_svm_ready improves over Exp06_ready

The disagreement pattern contains learnable information beyond raw confidence. The SVM router can identify when the regular model or cascaded model is more reliable.

### If Exp07+Aug improves over Exp07

LLM-generated training examples help the model generalize better, especially for rare or underrepresented entity labels.

### If Exp07+Aug does not improve

Possible explanations include:

1. generated examples are too noisy,
2. synthetic distribution differs from true evaluation distribution,
3. baseline training already has enough examples,
4. augmentation helps recall but hurts precision,
5. the model overfits generated patterns.

### If paired tests are significant

A significant paired test means the difference is consistent across seeds, not only caused by one lucky split.

The strongest evidence appears when both tests agree:

$$
p_{t-test} < 0.05
\quad\text{and}\quad
p_{Wilcoxon} < 0.05.
$$

---

## 21. Short Plain-English Summary

This pipeline is a rigorous experimental system for Hebrew NER. It tests four transformer models across multiple train/evaluation split strategies and augmented data variants. It compares a direct NER model, a cascaded three-step model, a structural consistency repair method, a confidence-based fusion method, and an SVM-based fusion method. Every comparison is repeated across 20 paired seeds so that improvements can be tested statistically rather than judged from a single lucky run.

The scientific architecture is therefore:

$$
\text{Data design}
\rightarrow
\text{Model training}
\rightarrow
\text{Cascaded decomposition}
\rightarrow
\text{Post-processing/fusion}
\rightarrow
\text{Entity-level evaluation}
\rightarrow
\text{Paired statistical inference}.
$$
