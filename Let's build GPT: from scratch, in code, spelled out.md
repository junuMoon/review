https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4948s

- Attention is a communication mechanism. Can be seen as a nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent wieghts.
- There is no notion of space. Attention simply acts over a set of vectors. This is why need to positionally encode tokens.
- Each example across batch dimension is of course prcoessed completely independent and never "talk" to each other
- In and "encoder" attention block just delete this single line that does masking with tril, allowing all tokens to communicate. On the other way, "decoder" block has triangular masking and is used in autoregressive settings.
- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention" the queries still get produced from x, but the keys and values come from some other, external source.
- "Scaled" attention additionaly divides `wei` by 1/sqrt(head_size). This makes it so when input Q, K are unit variance, `wei` will be unit variance too. And softmax will stay diffuse and not saturate too much. Softmax tends to sharpen towards the max.
- The Attention Block basically intersperses communication and the computation. The communicaition is done using multi-headed self-attention and then the computation is done using the feed fowrad network on all the tokens independently.
- Skip connections create a gradient superhighway that allows gradients to flow directly from the loss to the input, facilitating stable backpropagation.
  - Initially, residual blocks contribute minimally, but they gradually start to contribute more as optimization progresses.
- LayerNorm: 데이터의 feature중, 특정 feature의 분포가 너무 큰 경우는 Gradient Exploading 등 학습 중 장애가 우려되기에, batch normalization으로 한 샘플 내 feature 간 값의 차이를 줄여주는게 좋을 것 같다. 또는, 모델의 충분히 깊은 hidden layer 중, 특정 feature에 지나치게 의존해 Overfitting을 방지하고자 한다면, Layer Normalization을 통해 feature간 activation value 차이를 상쇄해줄 수도 있을 것 같다.
