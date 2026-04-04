# DP-SBR Notes

## 1. Paper Information

### Title
Data Poisoning Attacks to Session-Based Recommender Systems

### Authors
Yueqi Zhang, Ruiping Yin, Zhen Yang

### Venue
ICCNS 2022: Proceedings of the 2022 12th International Conference on Communication and Network Security

### Publication Information
Published in 2022 conference proceedings and listed as published in July 2023 in the ACM Digital Library. :contentReference[oaicite:0]{index=0}

---

## 2. Main Problem

This paper studies **data poisoning attacks against session-based recommender systems (SBR)**.

The attacker’s goal is to promote a **target item** so that it appears in the **top-K recommendation list for as many users as possible**. The paper emphasizes that, in SBR, the key challenge is not just choosing filler items, but also **arranging items within fake sessions** so that the fake sessions resemble realistic interaction patterns. :contentReference[oaicite:1]{index=1}

The paper positions itself as a systematic study of poisoning attacks on session-based recommender systems and proposes an attack framework to approximately solve the optimization problem behind fake-session construction. 

---

## 3. Why SBR Is Special in This Paper

The paper explains that session-based recommender systems rely on **user-item interaction sessions** rather than long-term historical user profiles. Because of that, the poisoning problem in SBR is centered on **constructing and arranging items in fake sessions**, rather than simply generating fake users with arbitrary ratings. :contentReference[oaicite:3]{index=3}

The paper uses **SR-GNN** as the representative target framework for session-based recommendation. It explicitly states that, without loss of generality, it focuses on SR-GNN and summarizes how SR-GNN models sessions as directed graphs and computes session representations for next-item recommendation. :contentReference[oaicite:4]{index=4}

---

## 4. Threat Model

The paper assumes the following threat model:

- the attacker wants the target item to appear in top-K recommendations for as many users as possible,
- the attacker can obtain user-item interaction information, especially timestamps,
- the attacker has limited resources and can only inject a limited number of fake users,
- each fake user can only interact with a limited number of items in one session,
- each injected session corresponds to a different fake user. :contentReference[oaicite:5]{index=5}

The paper denotes:

- `m` as the maximum number of fake users,
- `n` as the maximum number of filler items in a fake session. :contentReference[oaicite:6]{index=6}

---

## 5. Optimization Formulation

The paper formulates the poisoning attack as an optimization problem whose objective is to maximize the hit ratio of the target item. More concretely, it aims to maximize the proportion of users whose top-K recommendation list contains the target item. :contentReference[oaicite:7]{index=7}

The paper also points out three practical difficulties in solving this optimization directly:

1. the inputs are discrete variables such as item identities,
2. training the recommender is time-consuming,
3. fake sessions must preserve realistic interaction patterns to be effective. 

Because of these issues, the paper proposes an approximate attack framework rather than directly solving the full optimization exactly. :contentReference[oaicite:9]{index=9}

---

## 6. High-Level Attack Construction

The paper’s attack pipeline can be summarized as follows:

1. define fake-session parameters from the training-data distribution,
2. train a substitute model, called the **poison model**,
3. iteratively generate fake sessions using the poison model,
4. smooth the score distribution during generation,
5. replace one item in the generated session with the target item,
6. inject the fake sessions into the training set,
7. retrain and evaluate the attacked recommender. 

The paper’s Figure 1 and Algorithm 1 are the core implementation-oriented parts of the method. They describe the flow from training the poison model, to generating sessions, to finally inserting the target item through random replacement among top-k-percent positions. :contentReference[oaicite:11]{index=11}

---

## 7. Parameterized Fake Sessions

A key design choice in the paper is that fake sessions are **not generated fully from scratch without structure**. Instead, the paper first defines important fake-session parameters according to the statistics of the real training data. These include:

- the distribution of the **first item** in sessions,
- the distribution of **session lengths**. :contentReference[oaicite:12]{index=12}

### 7.1 Initial item distribution

The paper states that fake sessions are generated iteratively starting from an initial item, and that this initial item strongly affects the overall session pattern. Therefore, it controls the distribution of fake-session initial items to match the distribution of first items in the real training sessions. :contentReference[oaicite:13]{index=13}

### 7.2 Session-length distribution

The paper also controls the fake-session length distribution to match the real training sessions. It argues that session length contains important contextual information, including relations between items and their relative positions in the session. :contentReference[oaicite:14]{index=14}

---

## 8. Poison Model

The paper introduces a substitute model called the **poison model**.

Its role is to mimic the target recommender system and guide fake-session generation. The paper states that the poison model should be compatible with the target recommender system in:

- internal structure,
- hyperparameter setting,
- training dataset. :contentReference[oaicite:15]{index=15}

The attack construction then uses this poison model to generate fake sessions whose patterns resemble real sessions learned by the target system. The paper describes this as producing template-like sessions and then relating them to the target item. 

---

## 9. Fake Session Generation Process

After training the poison model, the paper generates each fake session iteratively.

The generation process is:

1. choose an initial item,
2. treat the current partial session as input to the poison model,
3. obtain a prediction score vector,
4. sample the next item from the first `K` items according to the processed scores,
5. append the sampled item to the session,
6. repeat until the predefined session length is reached. :contentReference[oaicite:17]{index=17}

This iterative generation is central to the paper’s method because it tries to preserve realistic session patterns rather than inserting arbitrary filler items.

---

## 10. Score Smoothing

The paper notes that real-world item interaction frequencies follow a **long-tail distribution**, meaning highly popular items may dominate the recommendation probabilities. To reduce this domination effect during fake-session generation, it applies **Min-Max Scaling** to the score vector. :contentReference[oaicite:18]{index=18}

The paper gives the Min-Max normalization formula and explains that the purpose is to make the score distribution smoother before sampling the next item. :contentReference[oaicite:19]{index=19}

---

## 11. Target Insertion Step

After generating the session, the paper performs the final poisoning step by **randomly replacing one item among the top `k%` positions of the constructed session with the target item**. This is the last step in Algorithm 1. :contentReference[oaicite:20]{index=20}

The paper later studies the impact of `k` and reports that the target item’s position matters significantly. It observes that larger `k` values tend to reduce attack effectiveness, and when `k` becomes too large, performance may even underperform the baseline attack. The paper attributes this to the relevance between item position and session pattern, and notes that earlier positions tend to be more influential. :contentReference[oaicite:21]{index=21}

---

## 12. Datasets

The paper evaluates the method on two real-world datasets:

- **Yoochoose**
- **Diginetica** :contentReference[oaicite:22]{index=22}

It states that, following prior work, it filters out:

- sessions of length 1,
- items appearing fewer than 5 times. :contentReference[oaicite:23]{index=23}

The paper further states that:

- after filtering, Yoochoose contains 7,981,580 sessions and 37,483 items,
- Diginetica contains 204,771 sessions and 43,097 items,
- for Yoochoose, it uses the most recent 1/64 portion of the training sessions, containing 369,859 sessions and 16,766 items. :contentReference[oaicite:24]{index=24}

---

## 13. Target Items

The paper evaluates two types of target items:

- **popular target items**
- **unpopular target items** :contentReference[oaicite:25]{index=25}

It describes the selection rule as:

- a **popular item** is an item clicked more than the average,
- an **unpopular item** is selected from items clicked fewer than 10 times in Yoochoose and Diginetica. :contentReference[oaicite:26]{index=26}

---

## 14. Evaluation Metric

The paper uses **P@K (Precision)** as the evaluation metric and specifically reports **P@25** in its main attack-effectiveness table. It defines P@K as the proportion of users whose top-K recommendation list contains the target item. :contentReference[oaicite:27]{index=27}

This means the evaluation is targeted: it is not measuring overall recommendation quality in the generic sense, but specifically how often the promoted target item appears in the recommendation list. :contentReference[oaicite:28]{index=28}

---

## 15. Baselines

The paper compares its method with two baseline attacks:

- **Random Attack**
- **AUSH (Attacking Recommender Systems with Augmented User Profiles)** :contentReference[oaicite:29]{index=29}

The paper describes Random Attack as generating filler items randomly, and AUSH as using reconstructed templates based on real user data. It argues that these baselines are weaker in SBR because they do not capture the input-session pattern as effectively as its method. 

---

## 16. Key Experimental Findings

### 16.1 Overall effectiveness

The paper reports that its attack is effective on both Yoochoose and Diginetica and outperforms the baseline attacks in the reported settings. For example, it reports that after inserting 1% fake users into Diginetica, the P@25 for popular target items increases by about 8 times relative to no attack. :contentReference[oaicite:31]{index=31}

### 16.2 Effect of attack size

The paper reports that larger fake-user injection size generally improves the attack effect. In its tables, performance increases as the fake-user ratio grows from 0.5% to 3% on both datasets. :contentReference[oaicite:32]{index=32}

### 16.3 Popular vs. unpopular targets

The paper states that the attack often has an even more significant effect on **unpopular items** than on popular items, because the injected fake sessions greatly increase the occurrence of an otherwise rare target item. :contentReference[oaicite:33]{index=33}

### 16.4 Effect of insertion-position range `k`

The paper reports that `k` has a large impact on effectiveness. As `k` increases, the P@25 value decreases, and when `k` is larger than 50, the method may underperform the baseline attack. The paper interprets this as evidence that item position inside the session matters for the poisoning effect. :contentReference[oaicite:34]{index=34}

---

## 17. Claimed Contributions

The paper explicitly claims three main contributions:

1. it provides the first systematic study on poisoning attacks to session-based recommender systems,
2. it formulates the attack as an optimization task and develops an approximate attack framework,
3. it evaluates the attack and compares it on real-world datasets. 

---

## 18. Conclusion of the Paper

The paper concludes that its poisoning attack method is more effective than the compared existing attacks for the evaluated settings. It summarizes the core idea as using a poison model to produce template sessions and relate them to target items to construct fake sessions for shilling. It also mentions future work in generating better template sessions and designing optimized attacks for other session-based recommender systems. :contentReference[oaicite:36]{index=36}

---

## 19. Stable Takeaways from the Paper

The most stable method-level takeaways from this paper are:

- SBR poisoning is centered on **fake-session construction** rather than only fake-user creation,
- realistic **session pattern preservation** is important,
- the paper relies on a **poison model** compatible with the target recommender,
- the method has a clear separation between:
  - parameterized fake-session generation,
  - iterative filler/template construction,
  - final target insertion,
- target-item **position** inside the session materially affects attack effectiveness. 