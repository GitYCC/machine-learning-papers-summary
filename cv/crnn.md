# An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (2015), Baoguang Shi et al.

###### contributors: [@GitYCC](https://github.com/GitYCC)

\[[paper](https://arxiv.org/pdf/1507.05717)\] \[[code](https://github.com/GitYCC/crnn-pytorch)\]

---

### Prerequisite

[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks (2006), Alex Graves et al.](speech/ctc.md)



### Introduction

- Compared with previous systems for scene text recognition, the proposed architecture possesses 4 distinctive properties:
  - end-to-end trainable
  - handles sequences in arbitrary lengths
  - both lexicon-free and lexicon-based scene text recognition tasks
  - smaller model
- The proposed neural network model is named as Convolutional Recurrent Neural Network (CRNN), since it is a combination of DCNN and RNN.



### The Proposed Network Architecture

- three components:
  - the convolutional layers
  - the recurrent layers
  - a transcription layer

![](assets/crnn_01.png)

#### Transcription

- Probability of label sequence

  - We adopt the conditional probability defined in the Connectionist Temporal Classification (CTC) layer proposed by [Graves et al](https://www.cs.toronto.edu/~graves/icml_2006.pdf).
  - $B$ maps $π$ onto $l$ by firstly removing the repeated labels, then removing the ’blank’s.
    - For example, $B$ maps `“--hh-e-l-ll-oo--”` (`-` represents blank) onto `“hello”`.
  - The conditional probability is defined as the sum of probabilities of all $π$ that are mapped by $B$ onto $l$:
    - $p(l|y)=\sum_{π:B(π)=l}p(π|y)$
    - $p(π|y)=\prod_{t=1}^{T}y^{t}_{π_t}$ 
    - $y^{t}_{π_t}$ is the probability of having label $π_t$ at time stamp $t$
    - Directly computing this eq. would be computationally infeasible due to the exponentially large number of summation items. However, this eq. can be efficiently computed using the forward-backward algorithm described in [Graves et al](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

- Lexicon-free transcription

  - $l^*\sim B(argmax_π p(π|y))$

- Lexicon-based transcription

  - $l^*=argmax_{l\in D}p(l|y)$  (where: $D$ is a lexicon)

  - However, for large lexicons, it would be very time-consuming to perform an exhaustive search over the lexicon.

  - This indicates that we can limit our search to the nearest-neighbor candidates $N_δ(l′)$, where $δ$ is the maximal edit distance and $l′$ is the sequence transcribed from $y$ in lexicon-free mode:
    $$
    l^*=argmax_{l\in N_δ(l′)}p(l|y)
    $$
    The candidates $N_δ(l′)$ can be found efficiently with the BK-tree data structure.

 