---
layout: post
title: RNN White Box
use_math: true
---

* Resources for learning RNN.  
* Backpropagation Through Time in detail.

LSTMs and GRUs, variants of vanilla RNNs, have proven to be extremely effective in Computer Vision and Natural Language Processing applications. Of all the excellent resources on RNNs available online, following are the ones that helped me understand them clearly:

1. [Understanding LSTM Networks:](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) Clear explanation with block diagrams.

2. [Recurrent Neural Networks Tutorial, Part 1 to Part 3:](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) Tutorial in Theano with a touch of BPTT and vanishing/exploding gradient problem.

3. [The Unreasonable Effectiveness of Recurrent Neural Networks:](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) Types of sequence problems and fun applications in NLP.

4. [Awesome Recurrent Neural Networks:](https://github.com/kjw0612/awesome-rnn) A curated list of RNN resources.

## Backpropagation Through Time

![](/images/VanillaNN.jpg  "Vanilla Neural Network")

'$Figure 1$' shows a vanilla neural network architecture. '$D$' dimensional input vector is conneted to the neural network unit through weights '$U$'. The neural network unit is inturn conneted to the '$K$' dimensional output vector through weights '$V$'. The neural network unit consists of a single layer of '$H$' neurons with '$tanh$' activation. Gradinets for optimzation of this network can be computed using standard back propagation.

RNNs, on the other hand, allow mapping between variable length input sequnces to variable length output sequences. Thus, gradients need to be backpropagated in time to update weights. Depending on the type of application, RNN sequences can be broadly categorized ([Refer Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)) into:

1. Synchronised input and output sequences

2. Unsynchronised input and output sequences

3. Input sequences

4. Output sequences

### Synchronised input and output sequences

![](/images/RNNSyncIpOp.jpg  "RNN architecture for synchronized input and output sequences")

Extending vanilla neural network architecture from '$Figure 1$', '$Figure 2$' shows a sequence of neural network units mapping the input sequences '$X_{i}$'s to output sequences '$Y_{i}$'s. Such architecture finds place in frame level video classification where the prediction depends on the current frame as well as the frames that appared before it. '$S_{i}$'s are the '$H$' dimensional output of the hidden layer in the neural network units. These are the memory of the network which transfer previous state information along the chain. The neural network unit at '$t+1$' takes input from '$X_{i}$' throught '$U$' and '$S_{t}$' through '$W$'. Weights '$U$','$V$' and '$W$' are shared across RNN units. '$S_{-1}$' is initialized to a vector of zeros.

The total loss for the above RNN is:

$$E = \sum_{t=0}^T E_t$$

where '$T$' is the length of input and output sequences. 
$\frac{\partial E}{\partial v_{hk}}$, $\frac{\partial E}{\partial u_{ij}}$, $\frac{\partial E}{\partial w_{dh}}$ are the target gradients that need to be computed.

#### Consider RNN unit at '$t = 0$' 

Compute the following

$$\frac{\partial E_{0}}{\partial y^k_{0}} \forall \tag{1}\label{1}$$

$$\frac{\partial E_{0}}{\partial S^h_{0}} = \sum_{k=1}^K \frac{\partial E_{0}}{\partial y^k_{0}}\frac{\partial y^k_{0}}{\partial S^h_{0}}\tag{2}\label{2}$$

$$\forall k \in \left{1,2,3...K\right}$$

$$\forall h \in \left{1,2,3...H\right}$$





