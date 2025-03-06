


# LLM RL without Tears

强化学习旨在解决一个**马尔可夫决策过程**，由一个五元组$\mathcal M=\langle\mathcal S,\mathcal A,\mathcal p,\mathcal r,\gamma\rangle$定义，其中$\mathcal S$是状态空间，$\mathcal A$是动作空间，$p:\mathcal S\times\mathcal S\times\mathcal A\to\mathbb R$是状态转移概率，$\mathcal r:\mathcal S\times\mathcal A\to\mathbb R$是奖励函数，$\gamma$是折扣因子。
从初始状态$s_0\in\mathcal S$开始，在每一个时刻$t\in\{0,1,\cdots,T-1\}$，代理观察到状态$s_t$，从策略分布$\pi:\mathcal A\times\mathcal S\to\mathbb R$采样动作$a_t\sim\pi(\cdot|s_t)$，环境产生奖励$r(s_t,a_t)\in\mathbb R$，并转移到下一个状态$s_{t+1}\sim p(\cdot|s_t,a_t)$，直到终止状态$S_T$。
代理的目标是优化策略$\pi$以极大期望累计奖励$J=\mathbb E_\pi\left[\sum_{t=0}^{T-1} \gamma^t r(s_t,a_t)\right]$。

LLM下的马尔可夫决策过程就是自回归生成的过程：状态空间$\mathcal S$是所有可能的前缀，动作空间$\mathcal A$是词表，状态转移是确定而非具有随机性的, *i.e.*, $s_{t+1}=\text{concat}(s_t,a_t)$，末尾为stop word, *e.g.*, [EOS], 的前缀为终止状态。
折扣因子$\gamma$通常设为1，即不对累积奖励设置时间步的衰减。
在此基础上，定义马尔可夫决策过程$\mathcal M$，或者准备强化学习训练，仅需[初始状态$s_0\in\mathcal S$的分布，即prompt的数据集](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)和[奖励函数](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)。

## 茴香豆之一：奖励函数的几种设置方法

LLM RL奖励函数的设计经历了以下阶段：

* 在[general RLHF](https://arxiv.org/pdf/1706.03741)中, OpenAI训练奖励模型为每一步生成（在LLM语境中为token-level）奖励, *e.g.*, $r_\phi(s_t,a_t)$。
* 在[LLM RLHF](https://arxiv.org/pdf/2009.01325)中，OpenAI转而训练奖励模型只为最终步生成outcome-level奖励, *e.g.*, $r_\phi(s_T)=r_\phi(s_{T-1},a_{T-1})$，而中间状态的奖励都是零。
这并不改变我们求解的问题：奖励只是累积到了最后而非每个步骤给出。这对奖励模型更加简单，因为其不用将奖励assign到各个token上，但需要强化学习算法本身完成credit assignment。
另外，为了缓解reward hacking问题，优化目标增加了和参考策略的**token-level** KL散度。OpenAI将其**加入了每个步骤的奖励中**。
* outcome-level奖励的稀疏性可能限制其性能，[PRM](https://openreview.net/pdf?id=v8L0pN6EOi)为每一个过程生成奖励，是token和outcome奖励的折衷。
* 在缓解reward hacking方面，[GRPO](https://arxiv.org/pdf/2402.03300)转向了**sequence-level** KL散度，并**放在了奖励之外**。另外，他们采用了方差较小的[k3估计](http://joschu.net/blog/kl-approx.html)。
这里的action/sequence-level，放在奖励内外和k1/2/3是三个相互正交的超参数，可以被任意组合。
* 在reasoning中训练一个准确的奖励模型是困难的，并且模型生成的答案可以被客观验证，因此[R1](https://arxiv.org/abs/2501.12948)移除了奖励模型，转向了基于规则的奖励。与此同时，由于不用考虑reward hacking，[KL正则也可以被去掉](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf)，这移除了参考模型。

## 茴香豆之二：模型的几种部署策略

LLM中的RL主流采用策略梯度方法，其核心是下述公式：

$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}[A^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s)]$$

这里的$A^{\pi_\theta}$称为advantage。直观地说，对给定状态$s$，$A^{\pi_\theta}(s,a)>0$的动作$a$具有正收益，应该提高其概率$\pi_\theta(a|s)$；$A^{\pi_\theta}(s,a)<0$的动作$a$具有负收益，应该降低其概率$\pi_\theta(a|s)$。
注意这里的期望是基于策略$\pi_\theta$的，这意味着状态-动作对$(s,a)$必须由当前的策略采得，这一类算法称为on-policy算法。
因此，策略梯度方法循环地执行如下流程：

* 从当前策略$\pi_\theta$ rollout得到状态-动作对数据$\{(s,a)\}$。
为了更高的throughput，这一过程通常使用vLLM等推理引擎。
* 为每一组状态-动作对$(s,a)$计算advantage $A^{\pi_\theta}(s,a)$。根据所使用的强化学习算法，这一过程可能包含奖励模型、价值模型和参考模型的推理。
三个模型的推理互不依赖，可以通过Ray调度在不同的设备上并行执行。
* 计算策略梯度$\nabla_\theta J(\theta)$，使用梯度上升更新策略参数，并将新的参数更新到推理引擎。如果使用的强化学习算法涉及价值模型，还要更新价值模型参数。
两个模型的训练互不依赖，可以通过Ray调度在不同的设备上并行执行。

和监督微调相比，强化学习效率的主要瓶颈是上述三个步骤的串行：
* 如果将不同的模型部署到不同的设备上，由于三个步骤的顺序依赖性，在进行一个步骤的计算时，部署其他模型的设备将闲置。
* 如果将所有模型部署到同一设备上，在执行一个步骤时，可能需要offload其他步骤的模型。
譬如在rollout时，可能需要offload其他模型以获得更多的memory和更高的throughput；在训练时，可能需要offload其他模型以有足够的memory缓存中间激活。
模型的on/offload带来了通讯成本。

不同的强化学习框架采用了不同的策略（见[表1](https://arxiv.org/pdf/2409.19256v2)）。我们可能希望更少地在步骤间切换以减小代价，譬如通过一次rollout更新多次模型。
在PPO中，重要性采样被用于实现这样的操作。

$$\begin{aligned}\nabla_\theta J(\theta)&=\mathbb E_{s\sim \pi_\theta,a\sim\pi_\theta(\cdot|s)}\left[A^{\pi_\theta}(s,a)\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}\right]\\&=\mathbb E_{s\sim\pi_\theta,a\sim\pi^{\text{old}}(\cdot|s)}\left[\frac{\cancel{\pi_\theta(a|s)}}{\pi^{\text{old}}(a|s)}A^{\pi_\theta}(s,a)\frac{\nabla_\theta\pi_\theta(a|s)}{\cancel{\pi_\theta(a|s)}}\right]\\&\approx\mathbb E_{\pi^{\text{old}}}\left[A^{\pi_\theta}(s,a)\frac{\nabla_\theta\pi_\theta(a|s)}{\pi^{\text{old}}(a|s)}\right]\end{aligned}$$

上式的约等号建立在$s\sim\pi_\theta$和$s\sim\pi^{\text{old}}$的近似性上。
这样，我们就可以通过$\pi^{\text{old}}$ rollout所得的状态-动作对数据$\{(s,a)\}$估算当前策略$\pi_\theta$的策略梯度。
记$r_\theta(s,a)=\pi_\theta(a|s)/\pi^{\text{old}}(a|s)$为新旧策略的比值，我们得到如下**代理目标**，它的梯度$\nabla_\theta L(\theta)$即为上式右侧。

$$L(\theta)=\mathbb E_{\pi^{\text{old}}}\left[A^{\pi_\theta}(s,a)r_\theta(s,a)\right]$$

我们不希望$\pi_\theta$和$\pi^{\text{old}}$的差异过大，否则$s\sim\pi_\theta$将不近似于$s\sim\pi^{\text{old}}$，影响估计的准确性。
PPO设计如下的代理目标，加入clipping来约束对当前策略$\pi_\theta$的更新，这实际上相当于**对部分的状态-动作对$(s,a)$施加了gradient masking**。

$$L^{\text{clip}}(\theta)=\mathbb E_{\pi^{\text{old}}}\left[\min(A^{\pi_\theta}(s,a)r_\theta(s,a), A^{\pi_\theta}(s,a)\text{clip}(r_\theta(s,a),1-\epsilon,1+\epsilon))\right]$$

注意对任意状态-动作对$(s,a)$，当$A^{\pi_\theta}(s,a)r_\theta(s,a)$较小时，$L^{\text{clip}}(\theta)$和$L(\theta)$相同；当$A^{\pi_\theta}(s,a)r_\theta(s,a)$较大时，**$\pi_\theta(a|s)$将不在$L^{\text{clip}}(\theta)$的计算图中，不参与反向传播，相当于在更新模型时排除了该状态-动作对$(s,a)$**。
那什么时候$A^{\pi_\theta}(s,a)r_\theta(s,a)$较大呢？
当$A^{\pi_\theta}(s,a)>0$，即动作$a$具有正收益时，当$r_\theta(s,a)>1+\epsilon$时，$A^{\pi_\theta}(s,a)r_\theta(s,a)$较大。
这表明从$\pi^{\text{old}}$到$\pi_\theta$的更新已经提高了动作$a$的概率，我们不希望其过度提高。
类似的，当$A^{\pi_\theta}(s,a)<0$，即动作$a$具有负收益时，当$r_\theta(s,a)<1-\epsilon$时，$A^{\pi_\theta}(s,a)r_\theta(s,a)$较大。
这表明从$\pi^{\text{old}}$到$\pi_\theta$的更新降低了动作$a$的概率，我们不希望其过度降低。


## 茴香豆之三：Advantage的几种算法

现在，强化学习pipeline只剩下最后一块拼图：advantage $A^\pi(s,a)$的计算。
Advantage的定义是动作价值函数和状态价值函数的差：$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$，其中

$$\left\{\begin{aligned}&Q^\pi(s,a)=\mathbb E_\pi\left[\sum_{l=t}^{T-1}r(s_l,a_l)\bigg|s_t=s,a_t=a\right]\\&V^\pi(s)=\mathbb E_\pi\left[\sum_{l=t}^{T-1}r(s_l,a_l)\bigg|s_t=s\right]\end{aligned}\right.$$

Advantage的计算主要有如下两个流派，前者需要训练价值网络而后者不需要：

* [GAE](https://arxiv.org/pdf/1506.02438)（有超参数$\lambda$，简单起见以[$\lambda=1$](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf)为例）:$A^\pi(s_t,a_t)=\sum_{l=t}^{T-1}r(s_l,a_l)-V_\phi(s_t)$，这里的$\sum_{l=t}^{T-1}r(s_l,a_l)$是$Q^\pi(s_t,a_t)$的Monte-Carlo估计，而$V_\phi(s_t)$是价值网络对$V^\pi(s_t)$的估计。
* [REINFORCE w/ baseline](https://aclanthology.org/2024.acl-long.662.pdf)：在计算advantage的时候把整个序列当做一个动作$y$，从prompt $x=s_0$经过一步到达终止状态，得到奖励$R(x,y)=\sum_{t=0}^{T-1}r(s_t,a_t)$。
据定义，此时有$Q^\pi(x,y)=R(x,y)$，$V^\pi(x)=\mathbb E_{y\sim\pi(\cdot|x)}[R(x,y)]$。
为了估计后者，我们在同一prompt $x$下采样多个completion $y_{1:n}\sim\pi(\cdot|x)$，计算其平均值$\text{mean}(R(x,y_{1:n}))$，因此有$A^\pi(x,y)=R(x,y)-\text{mean}(R(x,y_{1:n}))$。
此时策略梯度为$$A^{\pi_\theta}(x,y)\nabla_\theta\log\pi_\theta(y|x)=\sum_{t=0}^{T-1}A^{\pi_\theta}(x,y)\nabla_\theta\log\pi_\theta(a_t|s_t)$$和token-level的策略梯度相比较，这相当于令每个token的advantage为$A^{\pi_\theta}(s_t,a_t)=A^{\pi_\theta}(x,y)$。
[GRPO](https://arxiv.org/pdf/2402.03300)在此基础上，还除以了标准差：$A^{\pi_\theta}(s_t,a_t)=(R(x,y)-\text{mean}(R(x,y_{1:n})))/\text{std}(R(x,y_{1:n}))$。
REINFORCE-style算法的效果可能受限于其没有credit assignment以区分各个token的价值。

## 以veRL的超参数配置为例

以下是一些veRL中不太直观的超参数的意义：

* 关于rollout
    * `data.train_batch_size`：一次rollout使用多少prompt
    * `actor_rollout_ref.rollout.n`：每个prompt采多少completion

* 关于计算advantage
    * `algorithm.adv_estimator`：使用哪种方法计算advantage
    * `algorithm.kl_penalty`：如果将KL放在奖励中，使用k1/2/3估计

* 关于更新
    * `actor_rollout_ref.actor.ppo_mini_batch_size`：一次更新使用多少prompt，如果小于`data.train_batch_size`，那么一次rollout将更新多次，也就是使用了代理目标$L^{\text{clip}}$。实际更新所用的completion数还要乘`actor_rollout_ref.rollout.n`
    * `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`：一次正反向传播使用多少completion，这也确定了需要多少步梯度累积完成一次更新。建议使用`use_dynamic_bsz=True`和`ppo_max_token_len_per_gpu`实现packing
    * `actor_rollout_ref.actor.clip_ratio`：上述$L^{\text{clip}}$中的$\epsilon$
    * `use_kl_loss`：是否在奖励之外施加KL正则
    * `kl_loss_type`：如果将KL放在奖励外，使用k1/2/3估计
