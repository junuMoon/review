# Let's reproduce GPT-2 (124M) - Andrej Capathy

https://www.youtube.com/watch?v=l8pRSuU81PU&t=271s

- 임베딩 레이어란: 이산적인(discrete) 입력을 연속적인(continuous) 벡터 공간으로 변환하는 레이어
  - GPT 임베딩 레이어 = 토큰 임베딩 + 위치 임베딩
  - 토큰 임베딩: 각 기호에 고유한 ID를 부여. 이 ID를 기반으로 고정된 길이의 벡터로 변환. 학습을 통한 최적화.
- Dead RELU Problem: The “dead ReLU problem” occurs when neurons in a ReLU (Rectified Linear Unit) activated network only output zero. This happens when the weighted sum of the neuron’s inputs plus the bias term is less than or equal to zero, causing the ReLU function to output zero. As a result, the neuron stops learning, since the gradient during backpropagation is also zero. This can lead to significant portions of the network becoming inactive and not contributing to the model’s training.
- A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers
  - 가중치 초기화에서 1/root(N)으로 스케일링하여 뒤쪽 레이어의 학습 기울기를 작게 -> 기울기 폭발/소실 방지
- Loss Scaling: 낮은 정밀도에서 기울기 값에서 언더플로우가 발생하는 것을 방지. 손실 함수에 스케일 팩터를 곱해서 값을 키우고, 가중치를 업데이트하기 전에 다시 스케일로 나눠 값을 되돌리는 것.
- TF32
  - `torch.set_float32_matmul_precision('high')`: float32 matrix multiplications either use the TensorFloat32 datatype (10 mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers(approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication algorithms are available.  Otherwise float32 matrix multiplications are computed as if the precision is "highest".  See below for more information on the bfloat16 approach.

```
네, A100 Tensor Core의 TF32(Tensor Float 32)에 대해 자세히 설명해 드리겠습니다.

TF32는 NVIDIA가 A100 GPU에서 도입한 새로운 부동소수점 형식입니다. 이는 FP32의 정밀도와 FP16의 처리 속도 사이의 균형을 맞추기 위해 설계되었습니다.

## TF32의 구조

1. **총 비트 수**: 19비트
2. **부호 비트**: 1비트
3. **지수**: 8비트 (FP32와 동일)
4. **가수**: 10비트

## TF32의 특징

1. **FP32와 FP16의 하이브리드**:
   - FP32의 8비트 지수를 사용하여 넓은 동적 범위를 유지합니다.
   - 가수는 10비트로, FP32의 23비트보다는 적지만 FP16의 10비트와 동일합니다.

2. **내부 처리 방식**:
   - FP32 입력을 받아 내부적으로 TF32로 변환하여 연산을 수행합니다.
   - 결과는 다시 FP32로 변환되어 출력됩니다.

3. **성능 향상**:
   - FP32 대비 최대 8배의 연산 속도를 제공합니다.
   - 메모리 대역폭 사용량을 줄여 전체적인 시스템 성능을 향상시킵니다.

4. **정밀도와 속도의 균형**:
   - FP32보다 낮은 정밀도지만, 대부분의 딥러닝 워크로드에 충분한 정확도를 제공합니다.
   - FP16보다 넓은 동적 범위를 가져 수치적 안정성이 더 높습니다.

5. **자동 변환**:
   - CUDA 라이브러리와 딥러닝 프레임워크에서 자동으로 TF32를 사용하도록 설정할 수 있습니다.
   - 코드 수정 없이 성능 향상을 얻을 수 있습니다.

6. **유연성**:
   - 필요에 따라 TF32를 비활성화하고 전체 FP32 정밀도를 사용할 수 있습니다.

## TF32의 응용

1. **딥러닝 훈련**:
   - 대규모 신경망 모델의 훈련 속도를 크게 향상시킵니다.
   - 특히 컨볼루션 신경망(CNN)과 트랜스포머 모델에서 효과적입니다.

2. **추론**:
   - 실시간 추론 작업에서 높은 처리량을 제공합니다.

3. **과학 계산**:
   - 높은 정밀도가 필요하지 않은 과학적 시뮬레이션에서 사용될 수 있습니다.

## TF32의 한계

1. **정밀도 손실**:
   - FP32에 비해 정밀도가 낮아, 매우 정밀한 계산이 필요한 일부 응용 프로그램에는 적합하지 않을 수 있습니다.

2. **하드웨어 의존성**:
   - NVIDIA의 A100 및 이후 세대의 GPU에서만 사용 가능합니다.

TF32는 딥러닝 및 AI 워크로드에 최적화된 형식으로, 정밀도와 성능 사이의 균형을 잘 맞추고 있습니다. A100 GPU를 사용하는 대규모 AI 프로젝트에서 큰 성능 향상을 기대할 수 있으며, 특히 기존 FP32 코드를 거의 수정하지 않고도 이점을 얻을 수 있다는 점이 큰 장점입니다.
```

- loss, softmax layers are more susceptible to precision changes whethere major multiplies are more robust.
- `torch.compile`: Speedup mainly comes from reducing Python overhead and GPU read/writes, ...
  - w/o: step 80, loss: 5.921393871307373, dt: 423.49ms, tok/sec: 19344.00
  - w/ + compile, bf16 autocast: step 80, loss: 6.080526828765869, dt: 153.01ms, tok/sec: 53539.69
  - w/ + Flash Attention: step 80, loss: 5.925067901611328, dt: 138.27ms, tok/sec: 59245.93
    - https://github.com/ELS-RD/kernl/blob/main/tutorial/4%20-%20flash%20attention.ipynb
- cosine decay: we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375 million tokens.
- gradient clipping: 기울기 폭발을 방지하는 방법. 임계값이 넘어가면 기울기를 L2 norm으로 나눠 clipping을 해준다.

```python
def get_lr(step):
    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) if lr > lr_decay_iters, return min learning late
    if step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coef * (max_lr - min_lr)
```

- weight decay: All models use weight decay of 0.1 to provide a small amount of regularization.
  - L' = L + (λ/2) * ||w||^2
  - 기존 로스 값에 regulator를 추가하여 큰 가중치에 대하여 패널티를 줌.
  - only decaying embedding and matmul layers

```python
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")
        # Create AdamW optimizer and use the fuse version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
```

- gradient accumulation: Gradient를 n step 동안 누적 후 optimization step을 진행. 메모리가 적은 상황에서 큰 배치의 안정적인 학습을 대체하기 위함.
- Distributed Data Parallel: https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html
- multiprocessing.Pool.imap: A lazier version of map(). 결과를 준비되는 대로 하나씩 반환하는 이터레이터를 생성.
- multinomial: 다항분포. torch.multionomial은 input을 상대 확률로 보고 sample n개를 뽑음.
- `torch.gather`: https://www.youtube.com/watch?v=wEpnw_FDPu8 ... 머리 터진다
- `torch.unsqueeze`: 차원 수가 1인 차원을 지정한 dim에 생성
