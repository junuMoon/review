# Direct Policy Optimization
- https://arxiv.org/html/2305.18290v2
- to leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies.
- the DPO update increases the relative log probability of preferred to dispreferred responses, but it incorporates a dynamic, per-example importance weight that prevents the model degeneration that we find occurs with a naive probability ratio objective.
- Reward model loss
- Assuming access to a static dataset of comparisons $\mathcal{D} = \\{ x^{i} , y^{i}\_{w} , y^{i}\_{l} \\}^{N}\_{i=1}$ sampled from $p^*$, we can parametrize a reward model $r_φ(x,y)$ and estimate the parameters via maximum likelihood. Framing the problem as a binary classification we have the negative log-likelihood loss

$$
L\\{s_1\\}
$$

- DPO loss
- we have the probability of human preference data in terms of the optimal policy rather than the reward model, we can formulate a maximum likelihood objective for a parametrized policy $\pi\_{\theta}$. Analogous to the reward modeling approach (i.e. Eq. 2), our policy objective becomes:

$$
L
$$

- Gradient of DPO Loss
- The gradient of the loss function LDPO increases the likelihood of the preferred completions $y\_{w}$ and decreases the likelihood of dispreferred completions $y\_{l}$.
- Importantly, the examples are weighed by how much higher the implicit reward model $\hat{r}_{\theta}$ rates the dispreferred completions, scaled by $β$, i.e, how incorrectly the implicit reward model orders the completions, accounting for the strength of the KL constraint.

$$
L
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
