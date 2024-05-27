# Efficient Memory Management for Large Language Model Serving with PagedAttention
- https://arxiv.org/pdf/2309.06180
- For each request, this expensive process is repeated until the model out- puts a termination token. This sequential generation process makes the workload memory-bound, underutilizing the computation power of GPUs and limiting the serving throughput.
- For Transformers, these states consist of the key and value tensors associated with the attention mechanism, commonly referred to as KV cache, which represent the context from earlier tokens to generate new output tokens in sequence.

---

<img width="944" alt="image" src="https://github.com/junuMoon/review/assets/52732827/ce083207-8cc4-42e4-93b3-bbc33b926e69">

- Reserved Slots: are set aside for future use but are not currently holding any data.
- Internal Fragmentation: refers to unused memory within an allocated block.
  - To store the KV cache of a request in contiguous space, they pre-allocate a contigu- ous chunk of memory with the request’s maximum length (e.g., 2048 tokens).
- External Fragmentation: is the waste of memory between allocated blocks.
  - It occurs when there are small gaps between allocated memory blocks that are too small to be used for new allocations

---

- KV Cache: 새로운 토큰(단어)을 생성할 때 이전에 생성한 토큰들의 정보를 저장해 두는 공간입니다. 이 정보를 바탕으로 다음 토큰을 생성할 때 필요한 계산을 빠르게 할 수 있습니다.
- Q Cache가 없는 이유: 
  - Query 벡터는 현재 단어를 예측하기 위해 필요한 임시적인 계산입니다.
  - 반면, Key와 Value 벡터는 시퀀스의 각 토큰에 대해 고정된 정보로, 이후 토큰들을 생성할 때 반복적으로 사용됩니다. 따라서 Key와 Value 벡터는 캐시하는 것이 효율적입니다.
- 13B(130억) 파라미터를 가진 OPT 모델에서는 단일 토큰의 KV 캐시가 800KB의 공간을 차지
  - 벡터의 크기: 각 토큰에 대해 Key와 Value 벡터가 필요합니다.
  - 히든 스테이트 크기: 각 벡터는 5120 크기의 hidden state를 가집니다.
  - 레이어 수: 이 모델은 40개의 레이어를 가지고 있습니다.
  - FP16(16비트 부동 소수점): 각 숫자는 2 바이트를 사용합니다.
- 단일 토큰의 KV 캐시 크기: 2(key and value) * 5120(hidden state size) * 40(n_layers) * 2bytes(bytes per FP16) = 800kb
- 전체 KV 캐시 크기: 800 kb * 2048 tkns = 1.6gb

---

## Method
### Attention Score Calculation
$$
A_{ij} = \frac{\exp(q_i^T K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^T K_t / \sqrt{d})}
$$

- key block: $K_j = (k_{(j-1)B+1}, \ldots, k_{jB})$
  - $K_j$: j번째 블록의 키 벡터들의 집합.
  - $k_{(j-1)B+1}$: j번째 블록의 첫 번째 키 벡터.
  - $k_{jB}$: j번째 블록의 마지막 키 벡터.
  - $B$: 블록의 크기. 각 블록이 몇 개의 키 벡터를 포함하는지를 나타냄.
  - $j$: 현재 블록의 번호.

### KV Cache Manager

<img width="824" alt="image" src="https://github.com/junuMoon/review/assets/52732827/4d820508-1237-41ee-9aba-7aeefd471e7a">

- OS partitions memory into fixed-sized pages and maps user programs’ logical pages to physical pages. Contiguous logical pages can correspond to non-contiguous physical memory pages, allowing user programs to access memory as though it were contiguous.
- A request’s KV cache is represented as a series of logical KV blocks, filled from left to right as new tokens and their KV cache are generated. The last KV block’s unfilled positions are reserved for future generations. On GPU workers, a block engine allocates a contiguous chunk of GPU DRAM and divides it into physical KV blocks.
- The KV block manager also maintains block tables—the mapping between logical and physical KV blocks of each request. Each block table entry records the corresponding physical blocks of a logical block and the number of filled positions. Separating logical and physical KV blocks allows vLLM to dynamically grow the KV cache memory without reserving it for all positions in advance

### Scheduling and Preemption
- (1) Which blocks should it evict? (2) How to recover evicted blocks if needed again?
- Swapping: When vLLM exhausts free physical blocks for new tokens, it selects a set of sequences to evict and transfer their KV cache to the CPU. Once a request completes, its blocks are freed from memory, and the blocks of a preempted sequence are brought back in to continue the processing of that sequence.
- Recomputation: we simply recompute the KV cache when the preempted sequences are rescheduled. Note that recomputation latency can be significantly lower than the original latency, as the tokens generated at decoding can be concatenated with the original user prompt as a new prompt—their KV cache at all positions can be generated in one prompt phase iteration.

### Distributed Execution
- SPMD (Single Program Multiple Data) execution schedule, wherein the linear layers are partitioned to perform block-wise matrix multiplication, and the the GPUs constantly synchronize intermediate results via an all-reduce operation.
  - Specifically, the attention operator is split on the attention head dimension, each SPMD process takes care of a subset of attention heads in multi-head attention
  - vLLM에서의 모델 병렬 처리(Model Parallelism)는 멀티헤드 어텐션 블록의 각 헤드를 나누어 샤딩(Shard)하고, 각 샤드가 동일한 데이터를 처리하는 방식
- vLLM features a single KV cache manager within the cen- tralized scheduler. Different GPU workers share the manager, as well as the mapping from logical blocks to physical blocks. Although each GPU worker has the same physical block IDs, a worker only stores a portion of the KV cache for its corresponding attention heads.
  - the scheduler first prepares the message with input token IDs/block table
  - GPU workers start to execute the model with the input token IDs.
  - In the attention layers, the GPU workers read the KV cache according to the block table in the control message.
  - GPU workers send the sampled tokens of this iteration back to the scheduler

## Implementation

