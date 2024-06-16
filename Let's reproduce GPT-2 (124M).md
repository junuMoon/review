# Let's reproduce GPT-2 (124M) - Andrej Capathy

https://www.youtube.com/watch?v=l8pRSuU81PU&t=271s

![Image](file-service://file-4mnzLMfh8LokuV8z9O5rfniq)
- Dead RELU Problem: The “dead ReLU problem” occurs when neurons in a ReLU (Rectified Linear Unit) activated network only output zero. This happens when the weighted sum of the neuron’s inputs plus the bias term is less than or equal to zero, causing the ReLU function to output zero. As a result, the neuron stops learning, since the gradient during backpropagation is also zero. This can lead to significant portions of the network becoming inactive and not contributing to the model’s training.
