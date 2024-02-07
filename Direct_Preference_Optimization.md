# Direct Preference Optimization
- https://arxiv.org/html/2305.18290v2
- to leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies.
- the DPO update increases the relative log probability of preferred to dispreferred responses, but it incorporates a dynamic, per-example importance weight that prevents the model degeneration that we find occurs with a naive probability ratio objective.
- Reward model loss
- Assuming access to a static dataset of comparisons $\mathcal{D} = \\{ x^{i} , y^{i}\_{w} , y^{i}\_{l} \\}^{N}\_{i=1}$ sampled from $p^*$, we can parametrize a reward model $r_φ(x,y)$ and estimate the parameters via maximum likelihood. Framing the problem as a binary classification we have the negative log-likelihood loss

- RM model은 LM을 사용하고 헤드를 Regression head로 대체하여 스칼라 하나를 출력하게 하여 학습
    - `class trl.RewardTrainer`: The RewardTrainer can be used to train your custom Reward Model. It is a subclass of the transformers.Trainer class and inherits all of its attributes and methods. **It is recommended to use an AutoModelForSequenceClassification as the reward model**. The reward model should be trained on a dataset of paired examples, where each example is a tuple of two sequences. The reward model should be trained to predict which example in the pair is more relevant to the task at hand.


$$
\mathcal{L}\_R\left(r\_\phi, \mathcal{D}\right)=-\mathbb{E}\_{\left(x, y\_w, y\_l\right) \sim \mathcal{D}}\left[\log \sigma\left(r\_\phi\left(x, y\_w\right)-r\_\phi\left(x, y\_l\right)\right)\right]
$$

- RL Fine-Tuning Phase
<img width="550" alt="image" src="https://github.com/junuMoon/review/assets/52732827/8fecc76d-5773-44ec-b239-3671e2458a25">

- to express the reward function in terms of its corresponding optimal policy $\pi_r$, the reference policy $\pi_{ref}$, and the unknown partition function $Z(·)$.
<img width="374" alt="image" src="https://github.com/junuMoon/review/assets/52732827/cf826ce5-76d9-494c-b821-124d56dd319b">

- We can apply this reparameterization to the ground-truth reward $r_{∗}$ and corresponding optimal model $\pi_{*}$, using the Bradley-Terry model
- Because it depends only on the difference of rewards between two completions, i.e., $p{\_∗}(y1 ≻ y2 | x) = σ(r\_{∗}(x, y1) − r\_{∗}(x, y2))$
<img width="595" alt="image" src="https://github.com/junuMoon/review/assets/52732827/c17d62b3-1dc6-494c-85c9-852c493a8b83">

- DPO loss
- we have the probability of human preference data in terms of the optimal policy rather than the reward model, we can formulate a maximum likelihood objective for a parametrized policy $\pi\_{\theta}$. Analogous to the reward modeling approach (i.e. Eq. 2), our policy objective becomes:
- BT 모델은 preference score of y1 over y2
    - 위에서 도출한 r*를 BT 모델에 대입하여 reward 모델 없이 bt 모델을 정의 가능
    - 모든 (x, y_w, y_l)에 대하여 negative log prob을 구하여 DPO_loss를 계산할 수 있음

$$
\mathcal{L}\_{\mathrm{DPO}}\left(\pi\_\theta ; \pi\_{\text {ref }}\right)=-\mathbb{E}\_{\left(x, y\_w, y\_l\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi\_\theta\left(y\_w \mid x\right)}{\pi\_{\text {ref }}\left(y\_w \mid x\right)}-\beta \log \frac{\pi\_\theta\left(y\_l \mid x\right)}{\pi\_{\text {ref }}\left(y\_l \mid x\right)}\right)\right]
$$

- Gradient of DPO Loss
- The gradient of the loss function $L_{DPO}$ increases the likelihood of the preferred completions $y\_{w}$ and decreases the likelihood of dispreferred completions $y\_{l}$.
- Importantly, the examples are weighed by how much higher the implicit reward model $\hat{r}_{\theta}$ rates the dispreferred completions, scaled by $β$, i.e, how incorrectly the implicit reward model orders the completions, accounting for the strength of the KL constraint.

$$
\begin{aligned}
& \nabla\_\theta \mathcal{L}\_{\mathrm{DPO}}\left(\pi\_\theta ; \pi\_{\mathrm{ref}}\right)= \\
& -\beta \mathbb{E}\_{\left(x, y\_w, y\_l\right) \sim \mathcal{D}}[\sigma\left(\hat{r}\_\theta\left(x, y\_l\right)-\hat{r}\_\theta\left(x, y\_w\right)\right) \quad[\nabla\_\theta \log \pi\left(y\_w \mid x\right)-\nabla\_\theta \log \pi\left(y\_l \mid x\right)]]
\end{aligned}
$$

```python
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """
    pi_logps: policy logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    yw_idxs: preferred completion indices in [0, B-1], shape (T,)
    yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
    beta: temperature controlling strength of KL penalty

    Each pair of (yw_idxs[i], yl_idxs[i]) represents the indices of a single preference pair.
    """
    
    pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
    
    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps
    
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()

    return losses, rewards

```

- DPO outline: However, when $\pi_{SFT}$ is not available, we initialize $\pi_{ref}$ by maximizing likelihood of preferred completions $(x,y_w)$. This procedure helps mitigate the distribution shift between the true reference distribution which is unavailable, and $\pi_{ref}$ used by DPO.
    - If pi_ref is trained on (x, yw), and then trained by dpo on (x, yw, yl). We can think dpo as correction training? in korean term "오답노트"?
    - GPT4: So, when $\pi_{ref}$ is trained on the data of preferred completions (x, y_w), it is like a student learning the material for the first time. The subsequent training using DPO, which takes into account both the preferred completions (x, y_w) and the dispreferred completions (x, y_l), serves as a correction or an advanced learning step. It refines the policy's understanding based on additional insights about what is preferred and what is not, much like how a student would update their knowledge based on the corrections in their "오답노트".

