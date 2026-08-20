# Transformers

## The Transformer, From First Principles

A layer-by-layer walkthrough of the Transformer architecture from "Attention Is All You Need". Each section gives the intuition first, then the heavy math, then concrete technical examples. This document is the conceptual companion to a from-scratch implementation.

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [Input Embedding](#2-input-embedding)
3. [Positional Encoding](#3-positional-encoding)
4. [Self-Attention: Query, Key, Value](#4-self-attention-query-key-value)
5. [Scaled Dot-Product Attention](#5-scaled-dot-product-attention)
6. [Multi-Head Attention](#6-multi-head-attention)
7. [Add and Norm](#7-add-and-norm)
8. [Feed Forward Network](#8-feed-forward-network)
9. [The Complete Encoder Block](#9-the-complete-encoder-block)
10. [The Decoder](#10-the-decoder)
11. [Output Layer: Linear and Softmax](#11-output-layer-linear-and-softmax)
12. [The Whole Architecture](#12-the-whole-architecture)
13. [Notation Reference](#13-notation-reference)

---

## 1. The Core Problem

### Intuition

You have a sequence of tokens (words): "The cat sat on the mat." You want a model that understands each word in context. "Bank" means something different in "river bank" than in "money bank." The meaning of a token depends on the other tokens around it.

The older approach (RNNs) processed tokens one at a time, left to right, carrying a hidden state. This had two problems. It was sequential, so it could not be parallelized. And information from far-back tokens got diluted by the time processing reached the end, which is the long-range dependency problem.

The Transformer's bet is to throw away recurrence entirely and let every token look at every other token directly, all at once. That "looking at" mechanism is attention.

### What the rest of this document builds

Every layer that follows serves one of two purposes: it either mixes information across words (attention) or processes information within a single word (feed-forward). Everything else is supporting machinery that keeps a deep stack trainable.

---

## 2. Input Embedding

### Intuition

An embedding turns each word into a list of numbers that represents it. The computer cannot do math on the word "cat," so we assign every word a list of numbers:

```
cat  -> [0.2, -0.5, 0.8, 0.1, ...]
dog  -> [0.3, -0.4, 0.7, 0.2, ...]
car  -> [-0.9, 0.6, -0.1, 0.5, ...]
```

Each word gets its own list. That list is its embedding. The length of the list is a chosen number; in the original paper it is 512.

Why not give each word a single number (cat = 1, dog = 2)? A single number can only order things on one axis. With a list of numbers, a word can be similar to another word in some ways and different in others. A dog is close to a cat (both pets, both animals), but the numbers can also capture that one barks and one meows. One number cannot hold all that; a list can.

The phrase "dense vector in a continuous space where geometric closeness encodes meaning" unpacks like this:

- **Vector**: a list of numbers.
- **Dense**: most of the numbers are non-zero and meaningful (the opposite, sparse, would be mostly zeros).
- **Continuous space**: each list of numbers is a point in space. A list of 2 numbers is a point on a flat map, a list of 3 is a point in a room, a list of 512 is a point in a 512-dimensional space (impossible to picture, same idea).
- **Closeness encodes meaning**: words with similar meaning end up as points physically near each other.

The picture that makes it click, in 2D:

```
        pets
     cat . . dog

                    . car
                . truck
        vehicles
```

"Cat" and "dog" are near each other because they are similar. "Car" and "truck" are near each other and far from the pets. Distance between points now means difference in meaning. The trick is to turn meaning into geometry. These positions are learned during training; the model nudges the numbers until similar words drift together, because that arrangement helps it do its job.

### Math

You have a vocabulary of size $V$ and a chosen model dimension $d_{\text{model}}$ (512 in the paper). The embedding is a lookup table, a matrix:

$$E \in \mathbb{R}^{V \times d_{\text{model}}}$$

Read this as: $E$ is a table of real numbers with $V$ rows and $d_{\text{model}}$ columns. It is a spreadsheet with $V$ rows (one per word, e.g. 30,000) and $d_{\text{model}}$ columns (the 512 numbers for that word). To get "cat"'s embedding, go to cat's row and read off its 512 numbers.

- $E$ is the embedding table, the master dictionary holding every word's list.
- $\mathbb{R}$ means real numbers (values like 0.3, -1.7).
- $V$ is vocabulary size, how many different words the model knows.
- $d_{\text{model}}$ is the length of each word's list.

A token with index $i$ becomes the $i$-th row:

$$x_i = E[i] \in \mathbb{R}^{d_{\text{model}}}$$

For a sequence of $n$ tokens you get a matrix:

$$X \in \mathbb{R}^{n \times d_{\text{model}}}$$

Read this as: $X$ is a table with $n$ rows and $d_{\text{model}}$ columns. It is the actual input sentence after looking up each word.

- $n$ is the number of words in this specific sentence.
- $d_{\text{model}}$ is 512, same as before.

The difference between the two tables in one line: $E$ is the dictionary of all words; $X$ is just the specific words of your sentence pulled out of that dictionary.

### Technical Example

For the sentence "The cat sat" ($n = 3$), look up each word in $E$, grab its row of 512 numbers, and stack the three rows:

```
        (512 numbers each)
The  -> [ ...................... ]   row 1
cat  -> [ ...................... ]   row 2
sat  -> [ ...................... ]   row 3
```

That stack of 3 rows by 512 columns is $X$. The sentence is now fully in number form, ready for the next layer.

One detail from the paper: the embeddings are multiplied by $\sqrt{d_{\text{model}}}$ before positional encodings are added, to keep the scale of embeddings and positional signals comparable.

### The problem this creates

The embedding gives a set of vectors with no notion of order. "The cat sat" and "sat cat the" produce the same set of vectors. Since attention treats the input as an unordered set, word order is lost. That is what positional encoding fixes.

---

## 3. Positional Encoding

### Intuition

After embedding, the sentence is a stack of vectors, one per word, but those vectors carry no information about order. The word "cat" gets the same 512 numbers whether it is the 2nd word or the 20th, because its embedding is looked up purely by what the word is, never where it sits. To the model, these look identical:

```
"The cat sat"
"sat cat The"
"cat The sat"
```

Order obviously matters ("dog bites man" is not "man bites dog"), so we inject position information into each vector before attention sees it.

The idea: give each position (1st, 2nd, 3rd) its own distinct vector, a "position stamp," and add it onto the word's embedding.

$$\text{final vector for word} = \text{word embedding} + \text{position stamp}$$

So "cat" as the 2nd word becomes cat's meaning vector plus the position-2 stamp. The vector now encodes both what the word is and where it is.

There are two ways to make these stamps. You can learn them (another lookup table, one vector per position, trained like embeddings; many modern models do this). Or you can compute them with a fixed formula using sine and cosine waves, which is what the original paper does.

### Why sine waves

Good position stamps should have a few properties:

- Every position gets a unique stamp, no collisions.
- The pattern is smooth: position 5 and position 6 should have similar stamps (they are neighbors), while position 5 and position 50 are very different. Closeness in position means closeness in stamp.
- It generalizes to longer sequences than seen in training, because a formula keeps working at position 1000 even if you never trained that long.

Sine and cosine give all of this. Think of an analog clock. It represents any time using hands moving at different speeds: the second hand spins fast, the minute hand slower, the hour hand slowest. From the combined positions you read off a unique time. Positional encoding does the same with waves. Each of the 512 slots is a wave, and the waves run at different frequencies, some fast, some slow. A fast wave distinguishes neighboring positions finely; a slow wave captures the broad "roughly where in the sentence" sense. Read all 512 wave values together and you get a unique, smooth fingerprint for each position.

### Math

For position $pos$ (which word: 0, 1, 2, ...) and dimension $i$ (which of the 512 slots), the stamp is:

$$PE(pos, 2i) = \sin\!\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$

$$PE(pos, 2i+1) = \cos\!\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$

Decoding every piece:

- $pos$ is the position of the word (0 for first, 1 for second, ...).
- $i$ indexes the slot in the 512-long stamp. Even slots (0, 2, 4, ...) use $\sin$; odd slots (1, 3, 5, ...) use $\cos$. They are paired: slot 0 and slot 1 share a frequency, slot 2 and slot 3 share the next, and so on.
- $d_{\text{model}}$ is 512.
- $10000^{2i / d_{\text{model}}}$ is the frequency control. This term makes each pair of slots oscillate at a different speed.

Look at that denominator. When $i$ is small (early slots), the exponent $2i / d_{\text{model}}$ is near 0, so $10000^{\approx 0}$ is about 1. Dividing $pos$ by about 1 keeps the number large, so the sine wave cycles fast. When $i$ is large (later slots), the exponent approaches 1, so $10000^{\approx 1}$ is about 10000. Dividing $pos$ by about 10000 makes the number tiny, so the wave cycles very slowly.

```
slot 0,1     -> fast wave    (fine detail, distinguishes adjacent positions)
slot 2,3     -> slower
   ...
slot 510,511 -> slowest wave (coarse, broad position sense)
```

That is the clock: fast hands and slow hands, together giving a unique reading per position.

### Technical Example

Say $d_{\text{model}} = 4$ (instead of 512), so each stamp is 4 numbers. For the word at position $pos = 1$:

$$
\begin{aligned}
PE_1 &= [\, \sin(1 / 10000^0),\; \cos(1 / 10000^0),\; \sin(1 / 10000^{0.5}),\; \cos(1 / 10000^{0.5}) \,] \\
     &= [\, \sin(1),\; \cos(1),\; \sin(0.01),\; \cos(0.01) \,] \\
     &\approx [\, 0.84,\; 0.54,\; 0.01,\; 1.00 \,]
\end{aligned}
$$

Those 4 numbers are position 1's stamp. Position 2 gives a different set, position 3 different again, each unique, with neighbors staying close.

### Putting it together

The final input to the rest of the model:

$$X_{\text{final}} = X + PE$$

where $X$ is the stack of word embeddings ($n \times 512$) and $PE$ is the matching stack of position stamps ($n \times 512$). Same shape, added element by element. Each vector now says both "I am the word cat" and "I am in position 2." The $\sqrt{d_{\text{model}}}$ scaling is applied to $X$ before this addition so the word meaning is not drowned out by the position stamp.

---

## 4. Self-Attention: Query, Key, Value

### Intuition

Each word is now a vector that knows what it is and where it sits, but it still knows nothing about the other words. The vector for "bank" is identical in "river bank" and "money bank," when its meaning should shift depending on which word is nearby. We need a mechanism where each word can look at the other words and pull in relevant information to update its own understanding. That is attention. After it runs, "bank" next to "river" walks away with a vector that has absorbed some river-ness.

The cleanest way to understand attention is as information retrieval, like searching a database or looking things up in a library. You have a question, and a shelf of labeled folders. You state what you are looking for (your question), compare it against each folder's label to see which match, and pull the contents from the folders that matched best. Attention does exactly this, for every word, all at once. The three roles (question, label, content) are Query, Key, Value.

### Query, Key, Value

Every word produces three different vectors, each playing a distinct role:

- **Query (Q)**: "What am I looking for?" The word's question. "Bank" might ask "is there anything nearby that tells me which kind of bank I am?"
- **Key (K)**: "What do I offer? What am I about?" The word's label or advertisement, so others can decide if it is relevant. "River" broadcasts a key that says "I am about water and geography."
- **Value (V)**: "What information do I actually pass on if you pick me?" The content. If "river" gets attended to, its value vector, carrying its actual meaning, is what flows into "bank."

Why three separate vectors instead of using the embedding directly for everything: a word's role when asking is different from its role when being asked about, which is different again from what it contributes. Splitting into Q, K, V lets the word wear three different hats. "River" advertising itself (its key) can be tuned differently from the meaning it hands over (its value).

### Math

Each of Q, K, V is the word's vector multiplied by a learned weight matrix, three different matrices, one per role:

$$Q = X W^Q$$
$$K = X W^K$$
$$V = X W^V$$

$W^Q$, $W^K$, $W^V$ are learned during training. They are three different "lenses" the model looks at each word through: one extracts the question, one extracts the label, one extracts the content. Because they are learned, the model figures out on its own what makes a good question, a good label, and good content to pass along.

### The flow, intuition only

For a single word, say "bank," attention does this:

1. Take bank's Query, its question "what disambiguates me?"
2. Compare that Query against every word's Key (including its own). Each comparison produces a score: how well does this word's label answer bank's question? "River"'s key scores high; "the"'s key scores low.
3. Turn scores into weights that sum to 1 (a softmax, like converting raw match scores into percentages: river 70%, money 5%, the 2%).
4. Take a weighted blend of all the Values using those weights. High-scoring words contribute most of their value; low-scoring words contribute almost nothing.
5. That blend is bank's new vector, now infused with river-ness.

This happens for every word simultaneously, each with its own query, all as parallel matrix operations. No looping word by word like an RNN. That parallelism is why Transformers train fast.

### Why "self"

The queries, keys, and values all come from the same sequence; the words attend to each other within one sentence. Later, in the decoder, there is a variant where queries come from one sequence and keys and values from another, which is cross-attention.

---

## 5. Scaled Dot-Product Attention

### The equation we are building

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$$

Built in four steps: the dot product, the scaling, the softmax, and the final blend.

### Step 1: Compare query to key with the dot product

The operation that compares a query against a key is the dot product. Take two vectors, multiply element by element, add up the results, get a single number:

$$a \cdot b = a_1 b_1 + a_2 b_2 + \cdots + a_d b_d$$

Example:

$$[2, 1, 3] \cdot [1, 0, 2] = 2 \cdot 1 + 1 \cdot 0 + 3 \cdot 2 = 2 + 0 + 6 = 8$$

Why this measures match: the dot product is large and positive when two vectors point in a similar direction, near zero when they are unrelated (perpendicular), and negative when they point opposite ways. If "bank"'s query and "river"'s key point in similar directions, their dot product is big, a strong match. That single number is the relevance score.

$Q K^\top$ does all these comparisons at once. $Q$ and $K$ are matrices with one row per word:

- $Q$ is $n \times d_k$ (each of n words has a query of length $d_k$).
- $K$ is $n \times d_k$ (each word has a key of length $d_k$).

To get every query dotted with every key, multiply $Q$ by $K$ transposed ($K^\top$ flips it to $d_k \times n$ so shapes line up):

$$Q K^\top \in \mathbb{R}^{n \times n}$$

The result is an $n \times n$ grid, the score matrix. Row $i$, column $j$ is how much word $i$'s query matches word $j$'s key. For "The cat sat":

```
          The    cat    sat
   The  [  9      2      1  ]   <- The's query vs everyone's key
   cat  [  2      8      5  ]   <- cat's query vs everyone's key
   sat  [  1      5      7  ]   <- sat's query vs everyone's key
```

One row per word, telling you how much that word cares about each other word. This is the heart of attention; everything else is cleanup.

### Step 2: Scale by $\sqrt{d_k}$

Before doing anything with those scores, divide them all by $\sqrt{d_k}$, where $d_k$ is the length of the key and query vectors (64 in the paper).

The problem: a dot product sums up $d_k$ products. The longer the vectors, the more terms you add, so the sums naturally grow large in magnitude. With $d_k = 64$, scores can swing to big positive and negative values. This feeds into softmax next, and when softmax receives very large numbers it becomes extremely peaky, handing almost 100% of the weight to the single biggest score and about 0% to everything else. That is bad because the word ends up copying essentially one other word and ignoring useful context, and because it creates tiny gradients (softmax in its saturated, near-flat region barely responds to change), so learning stalls.

Dividing by $\sqrt{d_k}$ shrinks the scores back to a reasonable range, keeping softmax smooth and gradients healthy.

Why $\sqrt{d_k}$ specifically and not $d_k$: if query and key entries are independent with mean 0 and variance 1, then their dot product (a sum of $d_k$ such products) has variance $d_k$, which means a standard deviation of $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ renormalizes the standard deviation back to 1, undoing exactly the spread that summing introduced.

$$\frac{Q K^\top}{\sqrt{d_k}}$$

Same $n \times n$ grid, with tamed numbers.

### Step 3: Softmax turns scores into weights

The scores are raw numbers, some big, some negative. We want each row converted into weights that are all positive and sum to 1, so we can use them as blending proportions. The softmax formula for a row of scores $s_1, s_2, \ldots, s_n$:

$$\text{softmax}(s_j) = \frac{e^{s_j}}{\sum_{k} e^{s_k}}$$

Exponentiate every score (makes them all positive, and exaggerates gaps so bigger scores dominate), then divide each by the total so they add to 1.

Example, row of scores $[2, 1, 0]$:

$$
\begin{aligned}
\text{exponentiate:} &\quad e^2 = 7.39,\; e^1 = 2.72,\; e^0 = 1.00 \\
\text{sum:} &\quad 7.39 + 2.72 + 1.00 = 11.11 \\
\text{divide:} &\quad [0.67,\; 0.24,\; 0.09]
\end{aligned}
$$

Those add to 1. The word putting out these scores pulls 67% from the first word, 24% from the second, 9% from the third. Apply softmax row by row across the whole grid, so every word gets its own set of attention weights that sum to 1:

```
          The    cat    sat
   The  [ 0.90  0.07   0.03 ]
   cat  [ 0.10  0.65   0.25 ]
   sat  [ 0.05  0.35   0.60 ]
```

### Step 4: Multiply by V for the weighted blend

We have weights saying how much each word should pull from every other word. Now pull the content, the Values. $V$ is $n \times d_v$, one value vector per word. Multiply the weight matrix by $V$:

$$\underbrace{\text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right)}_{n \times n} \cdot \underbrace{V}_{n \times d_v} = \underbrace{\text{output}}_{n \times d_v}$$

For each word this takes a weighted average of all the value vectors using that word's attention weights. Take "cat"'s row $[0.10, 0.65, 0.25]$:

$$\text{new cat} = 0.10 \cdot V_{\text{The}} + 0.65 \cdot V_{\text{cat}} + 0.25 \cdot V_{\text{sat}}$$

Cat keeps mostly itself (65%) but blends in a chunk of "sat" (25%) and a little "The." The output is one new vector per word, same shape as we started (n rows), but every vector is now context-aware. "Bank" next to "river" has literally summed in a slice of river's value vector.

### The whole equation in plain English

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$$

- $Q K^\top$: every word compares its question against every word's label, giving a grid of match scores.
- $/ \sqrt{d_k}$: shrink the scores so softmax stays smooth and gradients stay healthy.
- $\text{softmax}$: turn each row of scores into blending weights that sum to 1.
- $\cdot\, V$: blend the content vectors by those weights, giving each word's new context-aware vector.

---

## 6. Multi-Head Attention

### Intuition

A single attention head can only focus on one kind of relationship at a time. When "cat"'s query blends everything into one weighted average, it produces one mixture. But a word relates to other words in many ways at once. Take "sat" in "The cat sat on the mat":

- Who sat? Relates to "cat" (subject).
- Where did it sit? Relates to "mat" (location).
- Grammatically, what agreement? Relates to different words again.

A single head has to cram all of these into one set of weights and one blended output. It is forced to compromise, averaging together relationships that are actually distinct, and you lose resolution.

The fix: run several attention operations in parallel, each with its own $W^Q$, $W^K$, $W^V$ matrices (its own lenses). Each is a head. Because each head has different learned matrices, each learns to focus on a different type of relationship:

```
Head 1  ->  might learn subject-verb links     (who did what)
Head 2  ->  might learn word-to-location links  (where)
Head 3  ->  might learn next-word / adjacency
Head 4  ->  might learn long-range references
   ...
```

Nobody tells the heads to specialize; they discover their specialties during training because varied lenses help the model. Then their outputs are combined, so the final result carries every kind of relationship at once.

### Dimensions

The paper uses $h = 8$ heads. Instead of giving each head the full $d_{\text{model}} = 512$ dimensions (which would be 8x the compute), the dimension is split across heads. Each head works in a smaller space:

$$d_k = d_v = \frac{d_{\text{model}}}{h} = \frac{512}{8} = 64$$

Each head operates on 64-dimensional queries, keys, and values. This is why $d_k = 64$ kept appearing; it is $512 / 8$. Because 8 heads times 64 dims equals 512 total, multi-head attention costs about the same as one full-size head but gives 8 specialized views instead of 1. You buy diversity essentially for free.

### Math

Step 1, each head does its own attention. For head $i$:

$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

where $\text{Attention}$ is the exact scaled-dot-product formula. Each head has its own projection matrices:

- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$: projects the input down to this head's 64-dim query space.
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$: same for keys.
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$: same for values.

Each head takes the full 512-dim input, projects it into its own 64-dim slice, and runs attention there. Output of each head: an $n \times 64$ matrix.

Step 2, concatenate the heads side by side:

$$\text{Concat}(\text{head}_1, \ldots, \text{head}_8) \in \mathbb{R}^{n \times 512}$$

Each head contributed 64 columns; 8 times 64 equals 512 columns back. For each word you have glued its 8 views into one long 512-vector.

Step 3, final projection. Multiply the concatenated result by one more learned matrix $W^O$:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

where $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ (512 x 512).

Why the final $W^O$: after concatenation the 8 heads' outputs are just stacked next to each other, sitting in separate 64-dim blocks, not yet talking to each other. $W^O$ mixes them together, letting the model blend information across heads into a single unified representation. Without it you would have 8 isolated views awkwardly glued; with it they are fused into one coherent output, shape $n \times 512$, ready for the next layer.

### Full picture

```
input X (n x 512)
   |
   |-> head 1: project to 64-dim -> attention -> (n x 64)
   |-> head 2: project to 64-dim -> attention -> (n x 64)
   |-> ...                                         ...
   |-> head 8: project to 64-dim -> attention -> (n x 64)
                                                    |
              concatenate all 8  ------------> (n x 512)
                                                    |
                          x W^O  ------------> (n x 512)  final output
```

### Full equation

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

Run $h$ separate attention operations in parallel, each with its own lenses looking at a different type of relationship; glue their outputs together; mix them with a final matrix into one unified representation.

---

## 7. Add and Norm

Two small operations bundled into one box: the Add (a residual connection) and the Norm (layer normalization). Both are about making deep networks trainable rather than adding representational power.

### The Add: residual connections

Transformers stack many layers (the paper uses 6 encoder blocks, each with attention plus feed-forward). Deep networks get harder to train as you stack: gradients shrink as they travel back through many layers (the vanishing gradient problem), so early layers barely learn. A deeper network can perform worse than a shallower one, not from lack of capacity, but because the signal cannot flow through it.

The fix is a shortcut. Instead of passing only the transformed output forward, add the original input back onto it:

$$\text{output} = x + \text{Sublayer}(x)$$

where $\text{Sublayer}(x)$ is whatever the block did (attention, or feed-forward). This is the arrow that curves around the block and rejoins at the Add and Norm.

What is literally happening:

```
x  ->  [ block ]  ->  Sublayer(x)
 |                        |
 |                        |
 +---------------------> (+)  ->  output
     (x skips around)
```

This helps two ways.

First, the block only has to learn the change. Without the shortcut, the block must output a complete finished vector, built from scratch. With the shortcut, the original vector $x$ is guaranteed to survive (it is added back), so the block only needs to produce the adjustment, the bit to add on top. Analogy: editing a document. Without residual, every editor hands you a fully rewritten document even if 95% is fine. With residual, each editor hands you a small list of changes, applied on top of the existing draft. Concretely: "bank" comes in with its base meaning; attention figures out to add river-ness; with the shortcut, attention only outputs the river-ness adjustment, and the base "bank" is preserved automatically. Bonus: if a block has nothing useful to add, it can output about zero, and then $\text{output} = x + 0 = x$, so the input passes through untouched. Without the shortcut, outputting zero would destroy the vector.

Second, gradients get a highway backward. When the network learns, the error signal (gradient) travels backward through every layer to update the early ones. In a deep stack it gets multiplied down at each layer and can shrink to near zero by the time it reaches early layers. The $+\, x$ fixes this: because the output is $x + \text{Sublayer}(x)$, the backward gradient hits that plus sign and splits into two paths. One goes through the block and gets weakened; the other flows straight through the $+\, x$ shortcut untouched, a direct wire from end to beginning. Picture a multi-floor building: without the shortcut a message from the top floor passes through every floor and gets garbled; the residual connection is an express elevator that rides straight down, clean. Even layer 1 in a very deep stack keeps receiving a strong signal.

### The adjustment, precisely

The word "adjustment" is concrete: it is a vector. The attention block outputs a 512-number vector, and that output vector is the adjustment. You add it, number by number, onto the input vector:

```
[ ... 512 numbers ... ]  +  [ ... 512 numbers ... ]  =  [ ... 512 numbers ... ]
   "bank" input              adjustment from attn         new "bank"
```

Where it comes from: attention produced a weighted blend of the other words' value vectors. In "river bank," attention decided bank should pull about 70% from river, so the output vector is mostly river's value vector. That blended vector, carrying river-ness, is the adjustment. Adding it onto bank's original vector gives a "bank" vector that now leans toward the water meaning. As a nudge in vector space:

```
        river meaning
             ^
             |  <- adjustment nudges bank this way
     bank o--+
   (before)
```

Bank starts at some point in the 512-dim space; the adjustment is an arrow that shifts that point toward the river region. So every Add is: old vector plus adjustment vector equals updated vector. The block's output is the adjustment; the "adjusted vector" is the result after adding the old one back.

### The Norm: layer normalization

As vectors flow through many layers of additions and multiplications, the numbers inside them can drift, growing large, shrinking, or becoming lopsided across scales. Unstable magnitudes make training erratic: a learning rate that works at one layer blows up or stalls at another. The fix is to normalize each vector after the add so its values sit in a consistent range: mean 0 and standard deviation 1, then a learned stretch and shift.

For one vector $x$ with components $x_1, \ldots, x_d$, compute its mean and variance across its own features:

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$$

$$\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$

Then normalize (subtract mean, divide by standard deviation):

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$\epsilon$ is a tiny number (like 1e-6) added to avoid dividing by zero. After this the vector is centered at 0 with spread 1. Finally, apply two learned parameters, a scale $\gamma$ and a shift $\beta$:

$$\text{LayerNorm}(x)_i = \gamma_i\, \hat{x}_i + \beta_i$$

Why $\gamma$ and $\beta$: forcing every vector to exactly mean 0, std 1 might be too rigid; sometimes the model wants a different scale or center for certain features. $\gamma$ (stretch) and $\beta$ (shift) are learned, so the model can adjust or undo the normalization if that helps. It gets stability and flexibility.

Important detail, layer norm not batch norm: normalization happens per word, across that word's own features. Each token's vector is normalized using only its own 512 numbers. It does not mix across different words or across the batch. This is what makes it work cleanly for variable-length sequences and is why it is called layer normalization.

### The whole box

$$\text{output} = \text{LayerNorm}\big( x + \text{Sublayer}(x) \big)$$

Read right to left: take the block's input $x$, add back the block's output (residual shortcut), then normalize to keep magnitudes stable.

```
        x --------------+  (residual shortcut)
        |               |
   Sublayer(x)          |
        |               |
        +---->  (+)  <--+   Add
              |
         LayerNorm          Norm
              |
           output
```

This box appears after every sub-layer: after each attention block and each feed-forward block. It is the connective tissue that makes the deep stack trainable.

### The clean split

- Attention and feed-forward do the thinking (mix context, transform meaning); they add representational power.
- Add (residual) keeps gradients and signal flowing through the deep stack.
- Norm (layer norm) keeps the numbers well-behaved so learning does not blow up or stall.

The last two are plumbing, not intelligence. Norm concretely buys consistent magnitudes across layers (the same learning rate works everywhere), smoother and more predictable gradients (no exploding or vanishing purely from size drift), and faster convergence (effort goes into learning meaning, not fighting unstable scales).

---

## 8. Feed Forward Network

### Intuition

Attention mixes information across words: "bank" pulls in river-ness, "sat" pulls in cat-ness. It moves information between positions. But at its core attention is just taking weighted averages of value vectors, a fairly linear blending operation. It is great at routing information around but limited in how much it can transform that information; it cannot do rich nonlinear processing on each word's content.

So after attention has gathered the right context into each word, you want a step that actually processes that gathered information: chew on it, transform it, extract higher-level features. That is the feed-forward network's job. The division of labor:

- Attention: gather the relevant information from other words (mixing across positions).
- Feed-Forward: think about what each word has gathered (processing within each position).

### Applied per word, independently

The feed-forward network is applied to each word's vector separately, with the same weights for every word. "Bank"'s vector goes through the FFN, "river"'s vector goes through the same FFN, "sat"'s through the same one, with no mixing between them (attention already did that). Each word is processed in isolation, like running the same little function over every item in a list. This is why it is called a position-wise feed-forward network.

### Math

Two linear layers with a nonlinearity (ReLU) in between:

$$\text{FFN}(x) = \max(0,\; x W_1 + b_1)\, W_2 + b_2$$

Step 1, expand. $x W_1 + b_1$ projects the vector up to a bigger dimension.

- $x$ is 512-dim.
- $W_1 \in \mathbb{R}^{512 \times 2048}$: expands 512 to 2048.
- $b_1$ is a learned bias vector added on.

The paper uses inner dimension $d_{ff} = 2048$, four times bigger than 512. A bigger space gives the network more room to pull apart and represent features, like spreading the information on a wider workbench so you can work on it more richly.

Step 2, nonlinearity. $\max(0, \cdot)$ is ReLU: replace every negative number with 0, keep positives as they are.

$$\text{ReLU}(z) = \max(0, z)$$

Example: $[3, -1, 0.5, -2] \rightarrow [3, 0, 0.5, 0]$.

Why this is the crucial part: without a nonlinearity, stacking linear layers is pointless, because two matrix multiplies in a row collapse into a single equivalent matrix multiply ($x W_1 W_2$ is just $x$ times some combined matrix). You gain nothing from depth. The ReLU breaks that linearity, letting the network learn genuinely complex nonlinear transformations. This is where the actual nonlinear thinking happens, and it is the reason the FFN adds representational power that attention alone cannot.

Step 3, contract. $(\cdot) W_2 + b_2$ projects back down to 512.

- $W_2 \in \mathbb{R}^{2048 \times 512}$: shrinks 2048 to 512.
- $b_2$ is another bias.

Back to 512 so the output matches the input shape (needed for the residual Add that follows). The shape journey:

$$512 \xrightarrow{\;W_1,\; \text{expand}\;} 2048 \xrightarrow{\;\text{ReLU}\;} 2048 \xrightarrow{\;W_2,\; \text{contract}\;} 512$$

Expand, apply nonlinearity, contract. That "expand, nonlinear, contract" sandwich is the signature of an FFN.

### Reading the FFN as key-value memory

There is a respected interpretation of the FFN as a pattern-matching memory:

- $W_1$ is the stored patterns. Each of the 2048 neurons in the expansion is a stored pattern. Computing $x W_1$ dots the token against every stored pattern (which patterns does this token match?). The ReLU zeros out non-matches, so only patterns that fire stay active.
- $W_2$ is the retrieved information. Each active neuron pulls its associated vector out of $W_2$ and contributes it.

Two precision notes on this reading. First, matching is soft, not discrete: each neuron fires by a continuous amount ($x W_1$ gives a real number, and ReLU keeps its actual magnitude), so a strong match fires hard and a weak match fires a little; it is a matter of degree, not yes/no. The output is a weighted sum of the associated vectors of all neurons that fired, each scaled by how strongly it fired, not a single retrieved item. Second, the FFN does not add anything back onto the token; it only outputs the blend. The adding back is the separate residual connection.

### Add and Norm again

The FFN's output is an adjustment, added back onto its input via the residual connection, then layer-normed:

$$\text{output} = \text{LayerNorm}\big( x + \text{FFN}(x) \big)$$

The same Add and Norm pattern: the FFN produces a per-word refinement, it gets added on, it gets normalized.

---

## 9. The Complete Encoder Block

Every piece of one encoder block:

```
        input (n x 512)
             |
    +--------+---------+
    |  Multi-Head Attn |   <- gather context across words
    +--------+---------+
         Add & Norm         <- x + attention adjustment, normalized
             |
    +--------+---------+
    |   Feed Forward   |   <- process each word's gathered info (nonlinear)
    +--------+---------+
         Add & Norm         <- x + FFN adjustment, normalized
             |
        output (n x 512)
```

Attention mixes across words, FFN thinks within each word, and each sub-layer is wrapped in Add and Norm. The paper stacks 6 of these blocks. Each block's output feeds the next, and the vectors get progressively richer and more contextual as they climb.

---

## 10. The Decoder

The decoder is built from pieces already covered: masked self-attention, cross-attention, a feed-forward network, and Add and Norm everywhere. Only two things are genuinely new, and both are variations on attention.

### What the decoder is for

The encoder read the input sentence and produced a rich, context-aware representation. The decoder generates the output one word at a time. For translation: the encoder reads English "The cat sat" and outputs vectors that understand it; the decoder produces the French translation word by word, using two sources of information:

1. What it has generated so far (the French words already produced).
2. The encoder's understanding of the input (the English).

That is why the decoder has two attention blocks stacked: one to look at its own output-so-far, and one to look back at the encoder.

### New piece 1: Masked Self-Attention

The bottom block is self-attention over the output sequence. Mechanically it is the exact scaled-dot-product multi-head attention already covered. The only difference is a mask.

The decoder generates words left to right: word 1, then word 2, then word 3. When producing word 3 it should only look at words 1 and 2, the words before it. It must not see words 4, 5, 6, because those do not exist yet at generation time. But during training we feed the whole target sentence in at once (for efficiency), so all words are physically present in the matrix. If we ran normal self-attention, word 3's query would attend to word 5's key, peeking at the future, at the answer it is supposed to predict. It would learn to cheat and then fail at real generation time.

The fix is to mask out the future. Before the softmax, set all future positions in the score matrix to negative infinity:

$$\text{scores}_{ij} = -\infty \quad \text{if } j > i \;\; (\text{word } j \text{ is after word } i)$$

Why negative infinity: the next step is softmax, and $e^{-\infty} = 0$. So any future position gets a weight of exactly zero; the word literally cannot pull information from words after it and is forced to rely only on itself and the past. The score grid for "Le chat ..." with the upper triangle (future) blanked:

```
         Le    chat   s'est
   Le  [  ok   -inf   -inf ]   <- "Le" sees only itself
  chat [  ok    ok    -inf ]   <- "chat" sees Le, chat
 s'est [  ok    ok     ok  ]   <- "s'est" sees all three (all in its past)
```

After softmax every negative infinity becomes 0. The formula gets one extra term, the mask $M$ (0 for allowed positions, negative infinity for future ones):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} + M \right) V$$

Same attention, plus a triangular mask that blindfolds each word to everything after it. This is why the decoder can be trained in parallel yet still generate honestly one word at a time.

### New piece 2: Cross-Attention (encoder-decoder attention)

The middle block is where the decoder looks back at the encoder's output, the bridge between the two towers. The crucial twist is where Q, K, V come from.

In all self-attention so far, Q, K, and V came from the same sequence. Here they come from two different places:

- Query comes from the decoder, from the word the decoder is currently working on. "As I generate this French word, here is what I am looking for."
- Key and Value come from the encoder's output, the understood English sentence. "Here are the English words' labels and content to match against and pull from."

$$Q = (\text{decoder state})\, W^Q$$
$$K = (\text{encoder output})\, W^K$$
$$V = (\text{encoder output})\, W^V$$

Intuitively, the decoder word asks "which parts of the English input are relevant to me right now?", matches its query against every English word's key, and pulls in the relevant English values. When generating the French word for "cat," this block lets the decoder focus on "cat" in the English source. This is how the model aligns input to output, how it knows which source words to translate at each step. Everything else (dot products, scaling, softmax, weighted blend) is identical to before; only the source of Q versus K and V differs. That is why it is called cross-attention: the query crosses over from the decoder to attend to the encoder.

### The full decoder block

```
        output-so-far (n x 512)
             |
    +--------+----------+
    | Masked Multi-Head |   <- look at words generated so far (no peeking ahead)
    |    Attention      |
    +--------+----------+
         Add & Norm
             |
    +--------+----------+
    |  Cross-Attention  |   <- look back at encoder's understanding
    |  Q=decoder,       |       (Q from decoder, K,V from encoder)
    |  K,V=encoder      |
    +--------+----------+
         Add & Norm
             |
    +--------+----------+
    |   Feed Forward    |   <- process, same as encoder's FFN
    +--------+----------+
         Add & Norm
             |
        output (n x 512)
```

Top to bottom: look at what I have said so far (masked self-attention), look at the source (cross-attention), think (FFN), each wrapped in Add and Norm. Like the encoder, this block is stacked 6 times.

---

## 11. Output Layer: Linear and Softmax

### Where we are

The decoder outputs a single 512-dim vector for the position it is predicting: a rich contextual summary of "given the input and everything generated so far, here is what I understand about the word that should come next." We need to convert 512 numbers into a choice among the vocabulary's roughly 30,000 words.

### The Linear layer

Project the 512-dim vector up to a $V$-dim vector, one number per vocabulary word:

$$\text{logits} = x W + b$$

- $x$ is the decoder's output, 512-dim.
- $W \in \mathbb{R}^{512 \times V}$ where $V$ is vocabulary size (about 30,000).
- Output: a $V$-dim vector, one score per word.

These raw scores are logits. Each logit says how strongly the model thinks that vocabulary word is the right next word. A high logit for "chat" means the model likes "chat" here; a low logit for "banana" means it does not.

```
512-dim understanding  --(Linear, W)-->  30,000 scores (logits)
                                          one per vocabulary word
```

Detail: this $W$ is often tied to the input embedding matrix $E$ from the start, the same table used to turn words into vectors, reused (transposed) to turn vectors back into word scores. It saves parameters and makes sense, since the same word-vector dictionary works both directions.

### The Softmax layer

The logits are raw scores, not interpretable as probabilities. Softmax converts them into a proper probability distribution over the vocabulary:

$$P(\text{word}_i) = \frac{e^{\text{logit}_i}}{\sum_{j} e^{\text{logit}_j}}$$

Exponentiate every logit, divide by the total, and now you have $V$ numbers that are all positive and sum to 1:

$$
\begin{aligned}
P(\text{"chat"}) &= 0.72 \\
P(\text{"chien"}) &= 0.11 \\
P(\text{"le"}) &= 0.04 \\
P(\text{"banana"}) &= 0.0001 \\
&\;\;\vdots \quad (\text{30,000 words, summing to 1})
\end{aligned}
$$

These are the Output Probabilities at the top of the diagram. The model is saying it is 72% confident the next word is "chat," 11% "chien," and so on.

### Picking the word and the loop

From this distribution you choose the next word; the simplest way is to take the highest-probability one. That word becomes the output for this position and gets fed back into the decoder's input as part of "generated so far" to predict the next word. This repeats, one word per step, until the model emits a special end-of-sentence token. That feedback loop is why the decoder input is labeled "Outputs (shifted right)": each generated word becomes part of the input for generating the following word.

```
decoder output  (512-dim vector)
        |
     Linear  -->  30,000 logits (raw score per word)
        |
    Softmax  -->  30,000 probabilities (sum to 1)
        |
   pick word  -->  "chat"  --+
        |                    | fed back in as input
        +--------------------+ to generate the next word
```

---

## 12. The Whole Architecture

Two towers: one understands, one generates, connected by cross-attention. Every layer either mixes information across words (attention) or processes within a word (feed-forward), wrapped in the plumbing (Add and Norm) that keeps the deep stack trainable.

Encoder (left tower), reads and understands the input:

1. Embedding: words to vectors (meaning as geometry).
2. Positional Encoding: stamp each vector with its position (waves).
3. Multi-Head Attention: every word gathers context from every other word (Q/K/V, dot-product, softmax, blend).
4. Add and Norm: residual shortcut plus stabilizing normalization.
5. Feed Forward: process each word's gathered info nonlinearly.
6. Stack 6 times, output a deep understanding of the input.

Decoder (right tower), generates the output one word at a time:

1. Masked Multi-Head Attention: look at words generated so far, never the future.
2. Cross-Attention: look back at the encoder's understanding (Q from decoder, K/V from encoder).
3. Feed Forward: process.
4. Stack 6 times.
5. Linear and Softmax: turn the final vector into next-word probabilities, pick a word, feed it back, repeat.

This is the same architecture under GPT, BERT, and essentially every modern LLM.

---

## 13. Notation Reference

| Symbol | Meaning |
| --- | --- |
| $V$ | vocabulary size (number of distinct tokens, e.g. ~30,000) |
| $d_{\text{model}}$ | model dimension, length of each token vector (512 in the paper) |
| $n$ | number of tokens in the current sequence |
| $h$ | number of attention heads (8 in the paper) |
| $d_k$ | dimension of query/key vectors per head ($d_{\text{model}} / h = 64$) |
| $d_v$ | dimension of value vectors per head (64) |
| $d_{ff}$ | inner dimension of the feed-forward network (2048) |
| $E$ | embedding table, $\mathbb{R}^{V \times d_{\text{model}}}$ |
| $X$ | sequence of token embeddings, $\mathbb{R}^{n \times d_{\text{model}}}$ |
| $Q, K, V$ | query, key, value matrices |
| $W^Q, W^K, W^V$ | learned projection matrices for Q, K, V |
| $W^O$ | output projection matrix in multi-head attention |
| $W_1, W_2$ | feed-forward weight matrices (expand, contract) |
| $b_1, b_2$ | feed-forward bias vectors |
| $\gamma, \beta$ | learned scale and shift in layer normalization |
| $\mu, \sigma^2$ | mean and variance in layer normalization |
| $\epsilon$ | small constant for numerical stability (e.g. 1e-6) |
| $M$ | attention mask (0 for allowed, $-\infty$ for future positions) |

### Standard hyperparameters (base model from the paper)

$$
\begin{aligned}
d_{\text{model}} &= 512 \\
h &= 8 \\
d_k = d_v &= 64 \\
d_{ff} &= 2048 \\
N &= 6 \quad (\text{number of encoder blocks and decoder blocks})
\end{aligned}
$$
