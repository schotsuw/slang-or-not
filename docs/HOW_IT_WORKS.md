# How It Works: From a Sentence to "Slang" or "Non-slang"

A visual guide for anyone new to this repo. For the full explanation see `PROJECT_EXPLAINED.md`; for implementation details see `PLAN.md`.

> The probabilities, vector numbers and "sits near" words in these diagrams are **illustrative**, to show the idea. Real values come from running the code.

---

## 1. The big picture

We give the system a sentence and tell it which word to look at. It answers whether that word is used as slang.

```mermaid
flowchart LR
    A["<b>Input</b><br/>Text: she's still salty about losing<br/>Target word: salty"] --> B["<b>Our model</b>"]
    B --> C["<b>Output</b><br/>slang"]
    D["<b>Input</b><br/>Text: the soup is too salty<br/>Target word: salty"] --> B
    B --> E["<b>Output</b><br/>non-slang"]
```

Same word, different context, different answer. The model has to use the **context** to decide.

---

## 2. Inside the main model (M1), step by step

This is our main approach: **frozen DistilBERT target-word embedding + logistic regression**.

```mermaid
flowchart TB
    S1["<b>1. Input</b><br/>she's still salty about losing<br/>target = salty"]
    S2["<b>2. Find the target word</b><br/>regex search in the raw text<br/>salty = characters 12 to 17"]
    S3["<b>3. Tokenize</b> (DistilBERT tokenizer)<br/>[CLS] · she · ' · s · still · salty · about · losing · [SEP]<br/>offsets tell us which token(s) cover characters 12 to 17"]
    S4["<b>4. Run DistilBERT</b> (frozen: not trained)<br/>reads the whole sentence and outputs<br/>one vector of 768 numbers per token"]
    S5["<b>5. Keep only the target's vector</b><br/>salty → [0.21, -1.04, 0.66, ... 768 numbers]<br/>(if salty was split into pieces, average them)"]
    S6["<b>6. Logistic regression</b><br/>trained on our labelled training data<br/>P(slang) = 0.91"]
    S7["<b>7. Output</b><br/>0.91 is above 0.5, so: slang"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
```

**Why step 5 matters.** DistilBERT gives *salty* a different vector depending on its neighbours, so the two sentences below produce different target-word vectors even though the word is spelled the same:

```mermaid
flowchart LR
    X1["she's still <b>salty</b> about losing"] --> V1["vector for salty<br/>(sits near: annoyed, bitter)"]
    X2["the soup is too <b>salty</b>"] --> V2["vector for salty<br/>(sits near: sweet, sour, spicy)"]
    V1 --> L["logistic regression<br/>learns which region = slang"]
    V2 --> L
    L --> O1["slang"]
    L --> O2["non-slang"]
```

---

## 3. The four models we compare

Every model gets the same input and gives the same kind of output. They differ in **what they look at**.

```mermaid
flowchart TB
    IN["<b>Same input</b><br/>she's still salty about losing · target = salty"]

    IN --> B0["<b>B0 · Most-frequent class</b><br/>ignores the text completely"]
    IN --> B1["<b>B1 · TF-IDF + logistic regression</b><br/>counts the words in the text"]
    IN --> M2["<b>M2 · DistilBERT whole-text + LR</b><br/>averages the vectors of ALL tokens"]
    IN --> M1["<b>M1 · DistilBERT target-word + LR</b><br/>uses ONLY the vector of salty"]

    B0 --> O0["always non-slang"]
    B1 --> OB1["slang or non-slang"]
    M2 --> OM2["slang or non-slang"]
    M1 --> OM1["slang or non-slang"]
```

| Model | What it looks at | Question it answers |
| --- | --- | --- |
| B0 | Nothing | What score do we get with no intelligence at all? |
| B1 | Which words appear | Are word counts enough? |
| M2 | Meaning of the whole text | Does a language model help? |
| M1 | Meaning of the target word in context | Does focusing on the target word help? (M1 vs M2 is our key comparison) |

---

## 4. The whole pipeline: data to results

```mermaid
flowchart TB
    RAW["<b>data/raw/</b><br/>SlangTrack CSV<br/>12,712 rows, 10 target words"]
    RAW --> KEEP["<b>Keep rows with text</b><br/>4,416 CCOHA passages (books and magazines, 1980–2010)<br/>tweet rows have no text in the public release"]
    KEEP --> SPLIT["<b>Split once</b> (stratified by word and label, seed 42)"]
    SPLIT --> TR["train.csv · 70%<br/>models learn from these"]
    SPLIT --> DV["dev.csv · 15%<br/>used to choose settings"]
    SPLIT --> TE["test.csv · 15%<br/>used once, for final numbers"]

    TR --> MODELS["<b>Models</b><br/>B0 · B1 · M1 · M2"]
    DV --> MODELS
    MODELS --> EVAL["<b>Evaluate on test.csv</b><br/>macro-F1 (main) · accuracy · precision · recall<br/>per-word scores · mistakes"]
    TE --> EVAL
    EVAL --> RES["<b>results/</b><br/>tables for the report"]
```

---

## 5. The held-out-word experiment

Can a model detect slang in a word it has **never seen** in training?

```mermaid
flowchart LR
    ALL["All 10 target words"] --> T8["<b>Train on 8 words</b><br/>chronic · eat · brownie · climber<br/>germ · cucumber · mammy · BMW"]
    ALL --> H2["<b>Hold out 2 words</b><br/>rodent · salty<br/>(never seen in training)"]
    T8 --> M["Train models"]
    M --> TEST["Test only on<br/>rodent and salty<br/>(547 passages, 122 slang)"]
    H2 --> TEST
    TEST --> Q["Does the model still<br/>recognize slang use?"]
```

We hold out *rodent* and *salty* because together they have 122 slang examples to test on. *Climber* was the first choice, but only 24 of its slang examples have text.

Example of what we're testing:

| Sentence | Correct answer |
| --- | --- |
| A **rodent** got into the kitchen. | non-slang |
| Don't trust him, he's a total **rodent**. | slang |

The model learned what slang *looks like* from the other 8 words. This test shows whether that knowledge transfers to new words, which matters because new slang appears all the time.
