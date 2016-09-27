---
layout: post
title: RNN White Box
---

* Resources for learning RNN.  
* Backpropagation Through Time in detail.

LSTMs and GRUs (varinats of vanilla RNNs) have proven to be extremely effective in Computer Vision and Natural Language Processing applications. Of all the excellent resources on RNNs available online, following are the ones that helped me understand them clearly:

1. [Understanding LSTM Networks:](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) Clear explanation with block diagrams.
2. [Recurrent Neural Networks Tutorial, Part 1 to Part 3:](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) Tutorial in Theano with a touch of BPTT and vanishing/exploding gradient problem.
3. [The Unreasonable Effectiveness of Recurrent Neural Networks:](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) Types of sequence problems and fun applications in NLP.
4. [Awesome Recurrent Neural Networks:](https://github.com/kjw0612/awesome-rnn) A curated list of RNN resources.

## Backpropagation Through Time (BPTT)

In this section I will explain BPTT wrt the types of RNN sequences a problem could be categorized into (refer [Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)).

Figure 1 shows the vanilla neural network architecture. The 'D' dimensional input vector is conneted to the neural network unit through weights 'U'. The neural network unit is inturn conneted to the 'K' dimensional output vector through weights 'V'. The neural network unit consists of a single layer of 'H' neurons with 'tanh' activation.

![alt text](https://github.com/sagarkurandwad/sagarkurandwad.github.io/blob/master/images/VanillaNN.png)




