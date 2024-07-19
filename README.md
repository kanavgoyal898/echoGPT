# echoGPT: character-level large language model

## Overview

**echoGPT** is a simplified implementation of a Generative Pre-trained Transformer (GPT) model designed for character-level language modeling with **25.4M parameters**. This implementation leverages PyTorch and includes features such as multi-head self-attention, feed-forward networks, and position embeddings. The model is trained on a text dataset (in this case, 'tiny_shakespeare.txt') to predict the next character in a sequence.

<div style="text-align: center;">
  <img src="./transformer.png" alt="Preview" style="width: 64%;">
</div>

## References

1. **Vaswani et al. (Google Research, 2017). [Attention is All You Need](https://arxiv.org/pdf/1706.03762)**: Introduced the transformer architecture, which utilizes self-attention mechanisms and parallel processing of input sequences, significantly improving the efficiency and scalability of deep learning models for NLP tasks.
2. **He et al. (Microsoft Research, 2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)**: Proposed residual networks (ResNets), which introduced skip connections to solve the vanishing gradient problem, enabling the training of very deep neural networks.
3. **Ba et al. (University of Toronto, 2016). [Layer Normalization](https://arxiv.org/pdf/1607.06450)**: Introduced layer normalization, a technique that normalizes the inputs across the features, stabilizing and speeding up the training of neural networks.
4. **Hinton et al. (University of Toronto, 2012). [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/pdf/1207.0580)**: Proposed the use of dropout, a regularization technique that prevents overfitting by randomly dropping units during training, improving the generalization of neural networks.

## Concepts

1. **Transformer Architecture**: 
   - Uses self-attention mechanisms to process input sequences in parallel, making it more efficient for long-range dependencies compared to RNNs.
   - Composed of multiple layers of self-attention and feedforward neural networks.

2. **Self-Attention**:
   - Allows the model to weigh the importance of different tokens in the input sequence when making predictions.
   - Implemented using multi-head attention, where multiple self-attention mechanisms run in parallel.

3. **Feedforward Networks**:
   - Each self-attention layer is followed by a feedforward network that processes the output of the self-attention mechanism.

4. **Layer Normalization**:
   - Applied to stabilize and speed up the training process.

5. **Positional Embeddings**:
   - Since transformers do not inherently understand the order of tokens, positional embeddings are added to token embeddings to encode the position of each token in the sequence.

## Hyperparameters

- **batch_size**: Number of sequences processed in parallel.
- **block_size**: Maximum context length for predictions.
- **max_iters**: Number of training iterations.
- **eval_intervals**: Interval for evaluating the model.
- **eval_iters**: Number of evaluation iterations.
- **learning_rate**: Learning rate for the optimizer.
- **fwd_mul**: Multiplier for the feedforward network's hidden layer size.
- **n_heads**: Number of attention heads.
- **n_layers**: Number of transformer layers.
- **n_embd**: Embedding size for tokens and positions.
- **dropout**: Dropout rate for regularization.

## Classes

### 1. `Head`
A single head of self-attention, which computes attention scores between tokens and produces weighted representations of values.

### 2. `MultiHeadAttention`
Combines multiple attention heads in parallel. Each head computes attention separately, and their outputs are concatenated and projected to form the final representation.

### 3. `FeedForward`
Defines a simple feed-forward neural network with one hidden layer, followed by a non-linear activation function (ReLU). This is applied independently to each token in the sequence.

### 4. `Block`
Represents a transformer block, consisting of multi-head self-attention followed by a feed-forward network. Layer normalization is applied after each sub-layer.

### 5. `BigramLanguageModel`
The main model class that encompasses the token and positional embedding layers, multiple transformer blocks, and the final output layer for character predictions. It includes methods for:
- Forward propagation (calculating logits and loss).
- Generating new text sequences based on a given context.

## Training

- **Data Preparation**: The text data is encoded into integers based on a character-level vocabulary. The dataset is split into training and testing sets.
- **Batch Generation**: Batches of input and target sequences are generated for training.
- **Loss Estimation**: The model's performance is periodically evaluated on the training and testing sets.
- **Optimization**: The AdamW optimizer is used to update the model parameters based on the computed gradients.

## Usage

1. **Prepare Data**: Ensure the `tiny_shakespeare.txt` file is in the `models` directory.
2. **Train the Model**: Run the training loop, which will print the training and testing losses at specified intervals.
3. **Generate Text**: After training, run `sample.py` to produce new text based on a given context.

## Requirements

- Python 3.x
- PyTorch (version compatible with CUDA if available)

## Sample Text

```
Et tu, Brute?

PRINCE:
My guiltle sovereign, of his name but the king;
Like tears then men af God's holy hands,
And live put up on this ireful spirit
As these breathed groan in all for a kiss.
Thou that goest more would, as thou dream of all
First in the cuares of receit and either room in
The weak of dignity, a kind of why, you know
To do a letter tyrant. What, if it be so?

MENENIUS:
Madam, this can you not her.

First Senator:
Unhappily use. For this time, a sigh gentleman.

MENENIUS:
My hope be ruled in thy capital;
Thy sepular's seat, if he did not go
But holp too.

VOLUMNIA:
If 't, thou goest, of good night, thy life
Of our city, an you whose scripe proceeding in children,
Where own wrinkled yet with lottless disposition,
I was the befit his wars; thou shalt be long
For being faintly now.

ISABELLA:
This I know my life; but I beseech you,
Because it is not.

ANGELO:
I promised!

ISABELLA:
And then I did!
Now for I must be order make his majesty
To pmon my death, dressolve the arguit of a dropper.`
```

## Statistics
```
tiny_shakespeare_2500.pt
step   500: train loss 1.8415, test loss 1.9685
step  1000: train loss 1.3468, test loss 1.5712
step  1500: train loss 1.2015, test loss 1.4930
step  2000: train loss 1.0994, test loss 1.4801
step  2500: train loss 0.9971, test loss 1.5281

tiny_shakespeare_5000.pt
step   500: train loss 1.8415, test loss 1.9685
step  1000: train loss 1.3468, test loss 1.5712
step  1500: train loss 1.2015, test loss 1.4930
step  2000: train loss 1.0994, test loss 1.4801
step  2500: train loss 0.9971, test loss 1.5281
step  3000: train loss 0.8902, test loss 1.5771
step  3500: train loss 0.7805, test loss 1.6503
step  4000: train loss 0.6549, test loss 1.7474
step  4500: train loss 0.5355, test loss 1.9054
step  5000: train loss 0.4320, test loss 2.0170
```

## Conclusion

echoGPT is a basic implementation of the transformer architecture, demonstrating key concepts in NLP and deep learning. It can be further enhanced by incorporating techniques such as learning rate scheduling, advanced sampling strategies, or more sophisticated data preprocessing.