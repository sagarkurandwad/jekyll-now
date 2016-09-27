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

The [types of RNN architecture](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) can be categorised into:

1. Synchronised Many-To-Many

2. Unsynchronised Many-To-Many

3. Many-To-One

4. One-To-Many

Befor we get into the details of these architectures, let us consider a vanilla neural network. 

![](/images/VanillaNN.jpg  "Vanilla Neural Network")

'Figure 1' shows a vanilla neural network architecture. 'D' dimensional input vector is conneted to the neural network unit through weights 'U'. The neural network unit is inturn conneted to the 'K' dimensional output vector through weights 'V'. The neural network unit consists of a single layer of 'H' neurons with 'tanh' activation. Gradinets for optimzation of this network can be computed using standard back propagation.








