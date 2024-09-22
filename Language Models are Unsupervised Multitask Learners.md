# Language Models are Unsupervised Multitask Learners

- [link](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)
- A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers
  - 가중치 초기화에서 1/root(N)으로 스케일링하여 뒤쪽 레이어의 학습 기울기를 작게 -> 기울기 폭발/소실 방지
    
