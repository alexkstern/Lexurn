# Experiment: Balls & Urns under Full Lex-Invariance

## Goal

Replicate the Balls & Urns generalization vs. memorization experiment with a Transformer that is **fully lex-invariant** — i.e., it receives a **new random token embedding for each sequence**

---

## Design

### Vocabulary

- m = 4 colors (red, green, blue, yellow)

### Urns (Tasks)

- Task is: infer the probability distribution of balls from some urn where the prob of each color from some specific urn is: [p(red),p(green),p(blue),p(yellow)]
- We have D number of tasks. We must generate D number of potential vectors that define the probability distribution over the colours of the balls in each of the D respective urn’s.
- D∈{4,8,16,32}

### Training Sequences

- At each training step:
    1. Sample **urn index** d∼Uniform(1,D)
    2. Draw 8 samples from a categorical distribution defined by whichever probability vector was sampled in step 1
    3. Apply a **new permutation to token embedding table, or simply do the assigning a random probability vector to each token ID like the lex invariant language modelling paper**
        
        → this changes token identities per sequence
        
    4. Use **causal language modelling CE loss** across all positions in this tensor

---

## Model Training (same as paper)

- Use a small Transformer decoder (4 layers, 4 heads, d_model = 128 or 256)
- N Training steps N∈{1000,5000,10000,20000}
- Use AdamW, LR = 1e-4, batch size = 64
- For each (D,N) combo, train 1 model

---

---

## 📊 Evaluation

### For each trained model (for each combination of N training steps and D tasks):

1. **Sample 256 test sequences**
    - These sequences are generated from the *same D urns* that were used during training.
2. **For each test sequence**:
    - Take the first 7 tokens as the **context**.
    - Compute the following:
        - **G(x)**: The generalizing predictor — just count how often each color appears in the context and normalize (i.e., turn counts into probabilities).
        - **M(x)**: The memorizing predictor — use Bayes' rule to combine all the urns seen in training for the model in question, and estimate which urn this test sequence likely came from.
            - You want to predict the next token by **guessing which urn (task) your current sequence came from**, and then **using that urn’s distribution** to make your prediction.
            - But since you're not sure which urn it was, you **average across all the urns**, weighted by how likely each one is to have produced the current context.
            - Multiply the probabilities of each token in the context according to this urn's color distribution.
    - Then compare the model's predicted distribution for the 8th token to both G(x) and M(x):
        - Compute how far the model is from G(x) → call this KL to G.
        - Compute how far the model is from M(x) → call this KL to M.
3. **Calculate the relative divergence**:
    - This is a single number between 0 and 1 that tells you **whether the model is acting more like a generalizer or a memorizer**.
    - If the number is close to **0**, the model behaves like G(x) → it generalizes.
    - If it’s close to **1**, the model behaves like M(x) → it memorizes.
    - If it’s around **0.5**, it’s somewhere in between.

---

### Heat Map

- Make a grid (heat map) where:
    - The **x-axis** is number of training steps (N)
    - The **y-axis** is number of tasks (D)
    - The **color** at each point shows the relative divergence value

**Expectation:**

If lex-invariance is working, the whole heat map should be **blue-ish** (i.e., values near 0), meaning the model **generalizes** in all cases and doesn't memorize.

---