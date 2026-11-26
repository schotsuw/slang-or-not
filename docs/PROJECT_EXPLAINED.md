# Our NLP Project, Explained

**Salty or Not? Detecting Slang Senses with Target-Word Embeddings**
CS4765/6765 Natural Language Processing, Fall 2026 · Saran Chotsuwan, Kristyn Le

This is a plain-language guide to what our project does and why. It is for us, the team. `PLAN.md` has the step-by-step implementation details.

---

## 1. The problem in one sentence

Many words have both a normal meaning and a slang meaning, and we want a model that can tell **which meaning is being used** in a given sentence.

## 2. Examples

| Sentence | Word | Meaning here | Label |
| --- | --- | --- | --- |
| The soup is too **salty**. | salty | tastes of salt | non-slang |
| She's still **salty** about losing. | salty | bitter, annoyed | slang |
| I **cooked** pasta for dinner. | cooked | prepared food | non-slang |
| We have a test tomorrow and we're **cooked**. | cooked | doomed, in trouble | slang |
| He has a **chronic** back problem. | chronic | long-lasting (medical) | non-slang |
| That party was **chronic**. | chronic | excellent | slang |
| A **rodent** got into the kitchen. | rodent | the animal | non-slang |
| Don't trust him, he's a **rodent**. | rodent | untrustworthy person | slang |

The word is spelled the same both times. **Only the surrounding words (the context) tell you the meaning.** That is what makes this hard for a computer.

## 3. Why this matters

- People read slang meanings instantly. NLP systems often don't, and take slang literally.
- That causes real errors:
  - **Content moderation:** "I'm dead 💀" (= that's hilarious) could be flagged as harmful.
  - **Machine translation:** "she's salty" translated word for word becomes "she contains salt".
  - **Sentiment analysis:** "that test was cooked" may be read as neutral or positive when it is negative.
- Slang changes fast. Senses like *cooked* (doomed), *ate* (did great) and *mid* (mediocre) spread only in the last few years, and a 2025 paper found that AI systems still read new slang literally (Ishita and Mamidi, 2025).

## 4. The task, precisely

This is **binary classification** (two possible answers).

- **Input:** a text (a passage from a book or magazine), plus a target word in it (e.g. text = "she's still salty about losing", target = *salty*).
- **Output:** `slang` or `non-slang`.

This is a version of **word sense disambiguation**, a classic NLP task: decide which meaning of an ambiguous word is meant. The course's project overview lists word sense disambiguation as one of the problems that can be framed as classification.

## 5. The dataset: SlangTrack

From the paper *The SlangTrack Dataset* (Aloraini et al., LChange 2026). Our instructor recommended it.

- The full dataset has **12,712 examples**, each containing one of **10 target words**:
  BMW, brownie, chronic, climber, cucumber, eat, germ, mammy, rodent, salty.
- But the public release leaves out the text of every tweet (it gives only tweet IDs), so only **4,416 examples have text**. Our instructor confirmed that using only these is a sensible approach, so **these 4,416 are our dataset**.
- Each example is labelled by human annotators as slang or non-slang. The annotators agreed strongly (Cohen's kappa 0.887).
- Our texts are **passages from books and magazines (CCOHA, the Clean Corpus of Historical American English), 1980–2010**. They are often long: 43 to 217 words, 91 on average.
- **Imbalanced:** about 83% non-slang (3,660) and 17% slang (756).
- Some words have very few slang examples: BMW (5), cucumber (9), germ (20) and climber (24). Their per-word scores won't be reliable.
- We split it ourselves into **train** (70%), **development (dev)** (15%) and **test** (15%):
  - **train:** the model learns from these
  - **development (dev):** we use these to choose settings (like how strong the regularization is)
  - **test:** used once at the end, for the final reported numbers

Watch out: the README's "48,508 unique words" is the vocabulary size (all different words in all the texts), not the number of target words.

Content warning: some target words have racist or sexual slang senses (*mammy*, *BMW*, *brownie*, *cucumber*), so some examples are offensive.

## 6. Key ideas you need first

**Vector / embedding.** A list of numbers that represents a piece of text, so a computer can work with it. Words or sentences with similar meanings get similar numbers. In NLP, "embedding" and "vector" mean basically the same thing here.

**Language model (BERT, DistilBERT).** A neural network trained on huge amounts of text that has learned how words relate to each other. **DistilBERT** is a smaller, faster version of Google's BERT that keeps most of its accuracy. It's free through the HuggingFace `transformers` library.

**Contextual embedding.** BERT-style models give each word a vector that depends on the **whole sentence**. So *salty* in "the soup is salty" and *salty* in "she's salty about losing" get **different** vectors. That is exactly what we need. (Older methods gave a word the same vector everywhere, which can't separate meanings.)

**Logistic regression.** A simple classifier we covered in class. It takes a list of numbers as input, learns a weight for each one, and outputs the probability of `slang`.

**Supervised learning.** The model learns from examples that already have the correct answer (our human labels). Our whole project is supervised.

## 7. Our approaches

We build several models and compare them. Each one answers a specific question.

### B0. Most-frequent-class baseline

- Always predicts `non-slang`, the most common label.
- No learning at all. It scores about **83% accuracy** while finding **zero** slang.
- **Why include it:** it's the floor. Any real model must beat it, and it shows why accuracy alone is misleading here.

### B1. TF-IDF + logistic regression (classic baseline)

- **TF-IDF** turns a text into word counts, weighted so that rarer, more informative words count more.
- Logistic regression then learns which words point to slang ("bro", "lol", "still") and which to non-slang ("soup", "recipe", "dinner").
- **Limits:** it ignores word order and meaning, and it only knows words it saw in training.
- **Why include it:** it answers "are word counts alone enough to detect slang?"

### M1. DistilBERT target-word embedding + logistic regression (our main approach)

This follows our instructor's feedback on the proposal: base the prediction on the embedding of **the target word itself**, not the whole text.

```
"she's still salty about losing"
        │
        ▼  1. find the target word in the text            →  "salty"
        ▼  2. run the whole text through DistilBERT        →  one vector per token
        ▼  3. take the vector(s) for "salty"               →  (if split into pieces like
        │                                                      "salt" + "##y", average them)
        ▼  4. one embedding of 768 numbers for "salty in THIS context"
        ▼  5. logistic regression
        ▼
     slang ✓
```

- DistilBERT is **not trained** here (it's "frozen"). We only use it to produce embeddings. That keeps things simple and fast.
- **Why this should work:** the target word's contextual embedding already encodes how the word is used in this sentence. Logistic regression just has to learn where the "slang" region of that space is.

### M2. DistilBERT whole-text embedding + logistic regression (comparison)

- Same as M1, but the embedding is the **average over all words** in the text instead of only the target word.
- **Why include it:** comparing M1 and M2 directly tests our instructor's point: **does focusing on the target word help?** This is the key comparison of the project.
- Example of where M2 could go wrong: in "lol my mom made brownies, she's so salty today", the whole-text average mixes in "brownies" and "lol". M1 looks only at *salty* (in context).

### M3. Fine-tuned DistilBERT with target markers (optional)

- Our original proposal: mark the word, like "she's still `<t>` salty `</t>` about losing", and fine-tune all of DistilBERT.
- Only if we have time left. Our instructor pointed out the markers may not make BERT focus on the word as intended, which is why M1 is now the main approach.

### Published benchmark

From the SlangTrack paper, fine-tuned BERT-large reached a **macro-F1 of 0.80**, or 0.85 with extra sentiment and emotion features. These used much bigger models and GPUs, and they also used the tweets, which we don't have. So they are a reference point, not a direct comparison. We don't need to beat them. They show where strong systems land, so we can say how close a simpler, cheaper approach gets.

## 8. Our experiments

### E1. Main comparison

Train all models on the same training set, test on the same test set, and compare.

Questions it answers:

- Do embeddings (M1, M2) beat word counts (B1)?
- Does the target-word embedding (M1) beat the whole-text embedding (M2)?
- How close do we get to the published results?

### E2. Held-out words: can a model handle words it has never seen?

New slang words appear all the time, so a useful model should handle words it was never trained on.

- Remove **every** example of *rodent* and *salty* from training.
- Train on the other 8 words.
- Test only on *rodent* and *salty* (122 slang examples together).
- We first planned to hold out *climber*, but it has only 24 slang examples with text, too few to test on.

What we expect: B1 should struggle, since it has never seen those words. M1 has a better chance, because DistilBERT learned from huge amounts of general text and its embeddings may still capture the "insult" or "figurative" use. But that's a prediction; the experiment tells us.

## 9. How we measure success

For each model we compute:

- **Accuracy:** % of predictions that are correct.
- **Precision (slang):** of everything the model called slang, how much really was slang?
- **Recall (slang):** of all the real slang, how much did the model find?
- **F1:** combines precision and recall into one number (their harmonic mean).
- **Macro-F1 (our main metric):** the F1 of the slang class and the F1 of the non-slang class, averaged equally.

**Why macro-F1, with a worked example.** Say the test set has 100 examples: 80 non-slang, 20 slang. The always-non-slang baseline (B0):

| Metric | Value |
| --- | --- |
| Accuracy | 80/100 = **80%** (looks good!) |
| Slang F1 | **0** (it never finds slang) |
| Non-slang F1 | 0.89 |
| **Macro-F1** | (0 + 0.89) / 2 = **0.44** |

Accuracy hides the failure; macro-F1 exposes it. It's also the metric the SlangTrack paper uses, so our numbers are comparable.

We also report **per-word results** (some words are much harder than others) and look at the model's **mistakes** to understand why it fails.

## 10. Constraints

- The whole project must run in **at most 1 hour** in the course environment (lab VM or a laptop with at most one GPU).
- Only use packages in the course environment: PyTorch, HuggingFace `transformers`, scikit-learn, numpy.
- Code runs on Mac, Windows and the course VM. Final numbers and timing come from one machine.

## 11. What we'll deliver

- **Implementation:** code that runs everything with one command and reproduces our results.
- **Report:** results tables (E1, E2, per-word), the M1 vs M2 comparison, error examples, and the ACL Responsible NLP checklist (offensive content in the data, possible unfair treatment of dialects such as African American English, compute used).
- **Presentation:** a fun demo slide could run the trained model on Gen Z examples like "we're cooked" vs "I cooked pasta".

## 12. Quick glossary

| Term | Meaning |
| --- | --- |
| Slang sense | The informal meaning of a word ("salty" = annoyed) |
| Target word | The word we're asking about in each example |
| Binary classification | Predicting one of two labels |
| Embedding / vector | A list of numbers representing text |
| Contextual embedding | A word's vector that depends on its sentence |
| DistilBERT | A small, fast, pretrained language model |
| Frozen | The model's weights are not changed; we only use its outputs |
| Fine-tuning | Training a pretrained model further on our task |
| TF-IDF | Word counts weighted by how informative each word is |
| Logistic regression | A simple classifier that outputs a probability |
| Baseline | A simple model that sets the bar to beat |
| Benchmark | Published results to compare against |
| Held-out words | Words removed from training to test generalization |
| Macro-F1 | F1 averaged equally over both classes |

## References

- Aloraini, A. M., Batista-Navarro, R., Nenadic, G., and Schlegel, V. (2026). The SlangTrack Dataset: Supporting the Detection of Words Used in Slang Senses. LChange 2026. https://aclanthology.org/2026.lchange-1.1/
- Ishita and Mamidi, R. (2025). The Evolution of Gen Alpha Slang: Linguistic Patterns and AI Translation Challenges. ACL 2025 Student Research Workshop. https://aclanthology.org/2025.acl-srw.43/
