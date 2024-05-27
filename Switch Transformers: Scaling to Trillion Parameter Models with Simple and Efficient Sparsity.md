# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- https://arxiv.org/pdf/2101.03961
- we instead propose a sparsely-activated expert model
- Scaling properties and a benchmark against the strongly tuned T5 model (Raffel et al., 2019) where we measure 7x+ pre-training speedups while still using the same FLOPS per token 
-	we investigate a fourth axis: increase the parameter count while keeping the floating point operations (FLOPs) per example constant. 
-	Our hypothesis is that the parameter count, independent of total computation performed, is a separately important axis on which to scale. 
-	우리의 가설은 수행된 총 연산량과 무관하게, 파라미터 수 자체가 모델 확장의 중요한 축이라는 것이다.
-	The gate-value for expert i is given by, 
<img width="285" alt="Pasted Graphic" src="https://github.com/junuMoon/review/assets/52732827/b9efa701-6d0d-4d0a-ae9d-014d4aa53432">

-	the output computation of the layer is the linearly weighted combination of each expert’s computation on the token by the gate value,
<img width="258" alt="Pasted Graphic 1" src="https://github.com/junuMoon/review/assets/52732827/ced4be7f-ddbd-49f0-8041-44968252aa33">

-	Distributed Switch Implementation. All of our tensor shapes are statically determined at compilation time, but our computation is dynamic due to the routing decisions at training and inference. Because of this, one important technical consideration is how to set the expert capacity. 
-	분산 스위치 구현. 모든 텐서의 모양은 컴파일 시점에 정적으로 결정된다. 그러나 스위치 트랜스포머에서의 연산은 학습 및 추론의 라우팅 결정으로 인해 동적으로 결정될 수 밖에 없다. 따라서 Expert Capacity의 설정값은 기술적으로 중요한 고려사항이다.
-	The expert capacity: the number of tokens each expert computes
<img width="671" alt="tokens per batch" src="https://github.com/junuMoon/review/assets/52732827/70ae9597-9733-4e01-950c-69363ce8ed0f">
  -	Switch Transformers perform better at lower capacity factors (1.0, 1.25). Smaller expert capacities are indicative of the scenario in the large model regime where model memory is very scarce and the capacity factor will want to be made as small as possible. 

---
## Perplexity
-	Perplexity는 언어 모델의 성능을 측정하는 지표 중 하나로, 모델이 실제로 관찰되는 시퀀스를 얼마나 잘 예측하는지를 나타냅니다. Perplexity는 모델이 특정 시퀀스를 생성할 확률의 역수를 취한 것으로, 값이 낮을수록 모델의 성능이 더 좋다고 평가됩니다.


```python
def calculate_perplexity(preds, targets):
    # preds는 각 단어에 대한 예측 확률이며, 각 행은 하나의 예측 확률 분포를 나타냅니다.
    # targets는 각 예측이 타겟하는 실제 단어의 인덱스를 나타냅니다.
    
    # 타깃 단어에 대한 확률을 추출합니다.
    target_probs = preds[np.arange(len(targets)), targets]
    
    # 각 타깃 단어의 로그 확률을 계산합니다.
    log_probs = np.log(target_probs)
    
    # 로그 확률의 평균을 계산합니다.
    avg_log_prob = np.mean(log_probs)
    
    # Perplexity를 계산합니다.
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity

# 예제 데이터
# preds는 각 단어에 대한 예측 확률 분포 (여기서는 간단한 예로 각 행은 하나의 예측에 해당)
# targets는 각 예측의 실제 타깃 단어 인덱스
preds = np.array([[0.1, 0.2, 0.7],
                  [0.8, 0.15, 0.05],
                  [0.3, 0.6, 0.1]])
targets = np.array([2, 0, 1])

# Perplexity 계산
perplexity = calculate_perplexity(preds, targets)
print("Calculated Perplexity:", perplexity)
```

- Switch Transformer 논문에서 NLP를 비교 메트릭으로 사용한 이유는 다음과 같습니다:
  1. NLP는 언어 모델의 성능을 직관적으로 이해할 수 있는 값입니다. NLP 값이 낮을수록 모델이 텍스트를 더 잘 예측한다는 것을 의미합니다.
  2. NLP는 크로스 엔트로피 손실과 직접적인 관련이 있습니다. 크로스 엔트로피 손실을 최소화하는 것은 NLP를 최소화하는 것과 같습니다. 따라서 NLP를 비교하는 것은 모델의 손실 함수를 직접 비교하는 것과 유사한 효과가 있습니다.
  3. NLP는 모델 간 비교에 공정한 지표입니다. 서로 다른 구조와 크기를 가진 모델 간에도 NLP를 직접 비교할 수 있습니다.
  4. NLP는 널리 사용되는 언어 모델 평가 메트릭 중 하나입니다. 다른 연구와의 비교를 용이하게 하기 위해 NLP를 사용했을 것입니다.

—---

-	The number of experts is the most efficient dimension for scaling our model. Increasing the experts keeps the computational cost approximately fixed since the model only selects one expert per token, regardless of the number of experts to choose from. The router must compute a probability distribution over more experts, however, this is a lightweight computation of cost O(d_model × num experts) where dmodel is the embedding dimension of tokens passed between the layers. 
-	We observe a clear trend: when keeping the FLOPS per token fixed, having more parameters (experts) speeds up training.
  -	FLOPS는 모델의 연산량을 측정하는 지표입니다. 토큰 당 FLOPS를 고정한다는 것은, 모델의 연산량을 일정하게 유지한 상태에서 전문가의 수만 변화시키는 것을 의미합니다.
<img width="1042" alt="Expert, Model and Dete" src="https://github.com/junuMoon/review/assets/52732827/9b7f2637-fc7b-4d15-b8dc-5c637aa3b939">
