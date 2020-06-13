# Reading Wikipedia to Answer Open-Domain Questions (2017), Danqi Chen et al.

###### contributors: [@GitYCC](https://github.com/GitYCC)

\[[paper](https://arxiv.org/abs/1704.00051)\] 

---

### Introduction

- Problem: answer to any factoid question from a text span in a Wikipedia article
- Our approach combines a search component based on bigram hashing and TF-IDF matching with a multi-layer recurrent neural network model trained to detect answers in Wikipedia paragraphs.
  - In order to answer any question, one must first retrieve the few relevant articles among more than 5 million items, and then scan them carefully to identify the answer.
- We develop DrQA, a strong system for question answering from Wikipedia composed of: 
  - (1) Document Retriever, a module using bigram hashing and TF-IDF matching designed to, given a question, efficiently return a subset of relevant articles
    - Our experiments show that Document Retriever outperforms the built-in Wikipedia search engine and that Document Reader reaches state-of-the- art results on the very competitive SQuAD benchmark
  - (2) Document Reader, a multi-layer recurrent neural network machine comprehension model trained to detect answer spans in those few returned documents.



### Our System: DrQA

- Document Retriever
  - Articles and questions are compared as TF-IDF weighted bag-of- word vectors.
  - We further improve our system by taking local word order into account with n-gram features.
    - Our best performing system uses bigram counts while preserving speed and memory efficiency by using the hashing of (Weinberger et al., 2009) to map the bigrams to 224 bins with an un-signed `murmur3` hash.
- Document Reader
  - Given a question $q$ consisting of $l$ tokens $\{q_1,...,q_l\}$ and a document or a small set of documents of $n$ paragraphs where a single paragraph $p$ consists of $m$ tokens $\{p_1, \dots , p_m\}$, we develop an RNN model that we apply to each paragraph in turn and then finally aggregate the predicted answers.
  - Question encoding:
    - apply BiLSTM to $\{q_1,...,q_l\}$ 's word embeddings
      - 300-dimensional Glove
      - We keep most of the pre-trained word embeddings fixed and only fine-tune the 1000 most frequent question words because the representations of some key words such as what, how, which, many could be crucial for QA systems.
    - and combine the resulting hidden units into one single vector: $q^*=\sum_jb_j\hat{q}_j$ (where: $b_j=exp(w\cdot q_j)/\sum_kexp(w\cdot q_k)$ and $w$ is a weight vector to learn)
  - Paragraph encoding
    - apply multi-layer BiLSTM to the feature vector $\tilde{p}_i$. The feature vector $\tilde{p}_i$ is comprised of the following parts:
    - Word Embeddings
      - 300-dimensional Glove
      - We keep most of the pre-trained word embeddings fixed and only fine-tune the 1000 most frequent question words because the representations of some key words such as what, how, which, many could be crucial for QA systems.
    - Exact Match: indicating whether $p_i$ can be exactly matched to one question word in $q$, either in its original, lowercase or lemma form
    - Token Features: include its part-of-speech (POS) and named entity recognition (NER) tags and its (normalized) term frequency (TF)
    - Aligned Question Embedding: soft alignments between similar but non-identical words
      - $f_{align}(p_i)=\sum_ja_{i,j}E(q_j)$
      - $E(q_j)$: Word Embeddings
      - $a_{i,j}=exp(\alpha(E(p_i))\cdot \alpha(E(q_j)))/\sum_k exp(\alpha(E(p_i))\cdot \alpha(E(q_k)))$
      - where: $\alpha(.)$ is a single dense layer with ReLU nonlinearity
  - Prediction
    - $P_{start}(i)\propto exp(p^*_iW_sq^*)$ and $P_{end}(i)\propto exp(p^*_iW_eq^*)$
    - During prediction, we choose the best span from token $i$ to token $i'$ such that $i\le i'\le i + 15$  and $P_{start}(i)\times P_{end}(i')$ is maximized

