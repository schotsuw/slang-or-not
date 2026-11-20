# slang-or-not

Course project for CS4765/6765 Natural Language Processing (Fall 2026).

Given a short text and a target word that has both a slang and a standard meaning, we predict whether the word is used as **slang** or **not slang** in that text.

- "the soup is **salty**" → not slang
- "she was **salty** after losing" → slang

We use the [SlangTrack dataset](https://github.com/SlangTrack/SlangTrack-ST-/) (Aloraini et al., 2026). It has 12,712 labelled examples covering 10 target words.

## Setup

You can run the project on the FCS lab computers (option 1) or on your own machine (option 2).

### Option 1: FCS lab computers

The course provides a ready-made environment, so you don't need to install anything.

1. Log in to an FCS lab computer, either [remotely](https://www.cs.unb.ca/help/remote-lab-gui-access.shtml) or in a lab room (e.g., GC112, ITD414/415) when no other course is using it.
2. Launch the NLP VM: **Applications → FCS VMs → NLP VM**.
3. In the NLP VM, open a terminal and activate the course environment:

   ```bash
   source nlpfall2026venv/bin/activate
   ```

4. Clone this repository and move into it:

   ```bash
   git clone https://github.com/schotsuw/slang-or-not.git
   cd slang-or-not
   ```

### Option 2: Your own machine

Requires Python 3.13.

```bash
git clone https://github.com/schotsuw/slang-or-not.git
cd slang-or-not
python3 -m venv nlpfall2026venv
source nlpfall2026venv/bin/activate
pip install -r requirements.txt
```

To activate the environment again later, run `source nlpfall2026venv/bin/activate` from inside the project folder.

### Data

The dataset is not included in this repository. Download it from the [SlangTrack GitHub repository](https://github.com/SlangTrack/SlangTrack-ST-/) and place it in the project folder.
