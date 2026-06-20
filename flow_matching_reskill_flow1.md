# Flow Matching ReSkill 流程说明

## 0. 变量定义

给定离线数据集：

$$
\mathcal{D}=\{(s_t,a_t,r_t,s_{t+1})\}
$$

从中切出长度为 \(H\) 的 chunk：

$$
S_t=(s_t,s_{t+1},...,s_{t+H-1})
$$

$$
A_t=(a_t,a_{t+1},...,a_{t+H-1})
$$

SkillVAE encoder 输出低维 skill latent：

$$
q_\phi(z|S_t,A_t)=\mathcal{N}(\mu_\phi(S_t,A_t),\sigma_\phi(S_t,A_t)^2)
$$

其中：

$$
z \in \mathbb{R}^{d_z}, \quad d_z \ll H d_a
$$

Flow skill prior 的 base noise 为：

$$
\eta_z \sim \mathcal{N}(0,I_{d_z})
$$

在线高层变量记为：

$$
c \in \mathbb{R}^{d_a}
$$

它可以来自两种来源：

1. 原 ReSkill / Flow ReSkill：高层 PPO 输出 \(n_t\)，此时 \(c_t=n_t\)。
2. 新版 condition flow policy：预训练流模型输出 \(c_t\sim p_\omega(c|s_t)\)。

当前实现里，离线训练 skill flow prior 使用：

$$
c^z = [s_t, \tilde{a}_t]
$$

在线采样 flow prior 使用：

$$
c^z_t = [o_t, c_t]
$$

其中 \(\tilde{a}_t\) 是带噪声的第一步动作。condition flow policy 的离线训练目标是：

$$
p_\omega(c|s_t) \approx p_{\mathcal{D}}(\tilde{a}_t|s_t)
$$

即用离线数据中的第一步动作近似可行的高层 condition。

## 1. 阶段一：训练 SkillVAE

SkillVAE 保持原 ReSkill 结构。

### 1.1 Skill encoding

对一个离线 chunk：

$$
(S_t,A_t)
$$

encoder 得到：

$$
\mu_z,\log\sigma_z = E_\phi(S_t,A_t)
$$

重参数采样：

$$
z = \mu_z + \sigma_z \odot \xi,\quad \xi \sim \mathcal{N}(0,I_{d_z})
$$

同一个 \(z\) 表示整个 chunk 的 skill。

### 1.2 Action decoder

decoder 仍然是原来的 SkillVAE decoder：

$$
\hat{a}_{t+i} = D_\theta(s_{t+i}, z), \quad i=0,...,H-1
$$

对应代码中在线执行时：

```python
obs_z = torch.cat((o_t, z), dim=1)
a_dec = skill_vae.decoder(obs_z)
```

### 1.3 SkillVAE loss

SkillVAE loss 仍然由重建项和 KL 项组成：

$$
\mathcal{L}_{skill} = \mathcal{L}_{BC} + \beta \mathcal{L}_{KL}
$$

其中：

$$
\mathcal{L}_{KL}
=D_{KL}(q_\phi(z|S_t,A_t)\|\mathcal{N}(0,I))
$$

代码位置：

```text
reskill/models/skill_vae.py
reskill/train_skill_modules.py
```

## 2. 阶段二：训练 Flow Matching Skill Prior

这一阶段替换原 ReSkill 中的 conditional RNVP prior。

原 RNVP 学的是：

$$
p_\psi(z|s_t,n_t)
$$

当前 flow matching prior 学的是一个条件速度场：

$$
v_\psi(z_\tau,\tau,c^z)
$$

其中条件为：

$$
c^z=[s_t,\tilde{a}_t]
$$

### 2.1 构造 skill prior 数据

先用当前 batch 的 SkillVAE encoder 得到目标 skill：

$$
z^* \leftarrow E_\phi(S_t,A_t)
$$

代码中使用的是 encoder 输出的 sampled latent：

```python
skill = output.z.detach()
```

条件状态使用 chunk 初始状态：

$$
s_t = S_t[0]
$$

条件动作使用 chunk 第一帧动作，并加入噪声：

$$
\tilde{a}_t = \frac{a_t}{2} + 0.2\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

于是 skill flow prior 的训练样本为：

$$
(c^z,z^*) = ([s_t,\tilde{a}_t],z^*)
$$

对应代码：

```python
state = data["obs"][:, 0, :]
action = data["actions"][:, 0, :] / 2.
action = action_ori + 0.2 * torch.normal(0, 1, action.shape)
condition = torch.cat([state, action], dim=1)
```

### 2.2 训练 condition flow policy

新版额外预训练一个 condition flow policy：

$$
p_\omega(c|s_t)
$$

它也是一个 flow matching 模型，但输入条件只有状态 \(s_t\)，输出空间是动作维度 \(d_a\)。离线训练目标不是 reward，而是拟合数据里的 noisy first action：

$$
c^* = \tilde{a}_t = \frac{a_t}{2}+0.2\epsilon
$$

因此训练样本为：

$$
(s_t,c^*)
$$

flow matching 形式为：

$$
x_0^c=\eta_c\sim\mathcal{N}(0,I_{d_a}),\quad x_1^c=c^*
$$

$$
x_\tau^c=(1-\tau)x_0^c+\tau x_1^c
$$

$$
v_\omega(x_\tau^c,\tau,s_t)\approx c^*-\eta_c
$$

loss 为：

$$
\mathcal{L}_{condition-FM}
=
\mathbb{E}
\left[
\left\|
v_\omega(x_\tau^c,\tau,s_t)-(c^*-\eta_c)
\right\|^2
\right]
$$

对应代码：

```text
reskill/train_skill_modules.py
reskill/models/bc_flow.py
```

关键点是：离线阶段只学习行为数据中的 \(p(c|s)\)，不使用 \(Q_c\)，也不做 RL 更新。

### 2.3 Flow matching prior loss

采样 skill-prior base noise：

$$
x_0^z = \eta_z \sim \mathcal{N}(0,I_{d_z})
$$

目标 skill：

$$
x_1^z = z^*
$$

采样 flow time：

$$
\tau \sim U(0,1)
$$

构造线性路径：

$$
x_\tau^z = (1-\tau)x_0^z + \tau x_1^z
$$

目标速度：

$$
u_\tau^z = x_1^z - x_0^z = z^* - \eta_z
$$

训练 teacher vector field：

$$
v_\psi(x_\tau^z,\tau,c^z) \approx z^*-\eta_z
$$

loss 为：

$$
\mathcal{L}_{prior-FM}
=
\mathbb{E}
\left[
\left\|
v_\psi(x_\tau^z,\tau,c^z)-(z^*-\eta_z)
\right\|^2
\right]
$$

对应代码：

```text
reskill/models/flow_prior.py::compute_flow_loss
```

### 2.4 Teacher Euler sampling

训练好 teacher 后，从噪声开始积分：

$$
z_0 = \eta_z
$$

每一步：

$$
z_{k+1}=z_k+\Delta t \cdot v_\psi(z_k,t_k,c^z)
$$

其中：

$$
\Delta t = \frac{1}{K}
$$

代码中默认：

```python
flow_steps = 10
```

对应代码：

```text
reskill/models/flow_prior.py::compute_flow_z
```

## 3. 阶段三：在线 ReSkill 训练

在线阶段有两种选择 condition 的方式。

### 3.1 原版：高层 PPO 输出 n

当前状态：

$$
o_t
$$

高层 PPO 输出：

$$
n_t = \pi_{high}(o_t)
$$

此时：

$$
c_t=n_t
$$

代码：

```python
n, v_agent, logp_agent, mu, std = agent.ac.step(o)
```

### 3.2 新版：condition flow policy 输出 c

如果开启：

```text
use_condition_flow = 1
```

则不再使用高层 PPO 输出 condition，而是用离线预训练的 condition flow policy：

$$
c_t \sim p_\omega(c|o_t)
$$

代码：

```python
c = condition_prior.sample_z_torch(o)
```

其中 \(c_t\) 的维度等于环境动作维度。它不是最终环境动作，而是 skill flow prior 的条件变量。

### 3.3 Flow prior 生成 skill latent

在线时没有真实第一步动作 \(a_t\)，所以使用在线选择出的 condition \(c_t\) 作为 skill flow condition 的第二部分。\(c_t\) 可以来自高层 PPO，也可以来自 condition flow policy：

$$
c^z_t = [o_t,c_t]
$$

然后：

$$
z_t = \text{FlowPrior}(c^z_t,\eta_z)
$$

如果 `use_student=0`：

```text
noise -> teacher Euler -> z
```

代码：

```python
cond = torch.cat((o, n), dim=1)
z = skill_prior.sample_z_torch(cond)
```

在 condition flow 版本中，上面代码里的 `n` 实际就是 condition flow policy 采样得到的 \(c_t\)。

## 4. 阶段四：执行 skill 与 residual 修正

拿到 \(z_t\) 后，在接下来的 \(H\) 个环境步中复用同一个 skill latent。

### 4.1 SkillVAE decoder 动作

每个环境步：

$$
a^{dec}_{t+i}=D_\theta(o_{t+i},z_t)
$$

代码：

```python
obs_z = torch.cat((o2, z), 1)
a_dec = skill_vae.decoder(obs_z)
```

### 4.2 Residual policy 修正

Residual PPO 输入：

$$
[o_{t+i}, z_t, a^{dec}_{t+i}]
$$

输出：

$$
a^{res}_{t+i}
$$

最终动作：

$$
a_{t+i}=a^{dec}_{t+i}+\alpha_{res}(k)a^{res}_{t+i}
$$

其中 residual 系数为：

$$
\alpha_{res}(k)=\frac{1}{1+\exp(-\kappa(k-C))}
$$

代码：

```python
residual_factor = logistic_fn(env_step_cnt, k=logistic_k, C=logistic_C)
a = a_dec + residual_factor * a_res
```

## 5. 阶段五：在线更新

### 5.1 原版高层 PPO 更新

高层 PPO 的 action 是：

$$
n_t
$$

高层 reward 是整个 skill chunk 的累计 reward：

$$
R^{skill}_t=\sum_{i=0}^{H-1} r_{t+i}
$$

所以高层 PPO 学：

$$
\pi_{high}(o_t)\rightarrow n_t
$$

目标是让 `n` 经过 flow prior 后生成更好的 skill latent \(z\)。

### 5.2 Condition flow policy 版本

当 `use_condition_flow=1` 时，高层 PPO 不再更新，condition flow policy 本身也不通过 RL 反向更新参数。在线改进来自一个 condition critic：

$$
Q_c(o,c)
$$

它评价在状态 \(o\) 下选择 condition \(c\)，再经过 skill flow prior 采样 \(z\)、decoder 执行一个 skill chunk 后的长期价值。

在线收集 transition：

$$
(o_t,c_t,R^{chunk}_t,o_{t+H},done)
$$

其中：

$$
R^{chunk}_t=\sum_{i=0}^{H-1}\gamma^i r_{t+i}
$$

当前实现使用 TD target 训练 \(Q_c\)：

$$
y_c
=
R^{chunk}_t
+
(1-done)\gamma^H
\min_j Q^{target}_{c,j}(o_{t+H},c_{t+H})
$$

其中：

$$
c_{t+H}\sim p_\omega(c|o_{t+H})
$$

critic loss：

$$
\mathcal{L}_{Q_c}
=
\mathbb{E}
\left[
\left(Q_c(o_t,c_t)-y_c\right)^2
\right]
$$

对应代码：

```text
reskill/rl/agents/chunk_critic.py::update_with_condition_policy
```

### 5.3 Residual PPO 更新

Residual PPO 每个环境步更新，状态为：

$$
[o_{t+i}, z_t, a^{dec}_{t+i}]
$$

动作是：

$$
a^{res}_{t+i}
$$

reward 是环境单步 reward：

$$
r_{t+i}
$$

## 6. Q-guidance Flow Sampling

当前有两类 guidance。

### 6.1 Skill latent guidance: \(Q_z(o,z)\)

如果开启：

```text
use_grad=1
guidance_scale > 0
use_student=0
```

则在线训练一个 chunk critic：

$$
Q_z(o,z)
$$

它学习 skill chunk 的折扣回报：

$$
R^{chunk}_t = \sum_{i=0}^{H-1}\gamma^i r_{t+i}
$$

在 teacher Euler sampling 中，每一步额外加入 Q 对 \(z\) 的梯度：

$$
z_{k+1}
=
z_k
+
\Delta t
\left(
v_\psi(z_k,t_k,c^z)
+
\lambda_z \nabla_z Q_z(o,z_k)
\right)
$$

其中：

```text
lambda_z = guidance_scale
```

代码：

```text
reskill/models/flow_prior.py::compute_flow_z_guided
reskill/rl/agents/chunk_critic.py
```

这类 guidance 直接调整 skill latent \(z\)，前提是使用 teacher Euler sampling：

```text
use_student = 0
```

### 6.2 Condition guidance: \(Q_c(o,c)\)

如果开启：

```text
use_condition_flow = 1
condition_use_grad = 1
condition_guidance_scale > 0
use_student = 0
```

则 condition flow policy 在采样 \(c\) 时也加入 critic gradient：

$$
c_{k+1}
=
c_k
+
\Delta t
\left(
v_\omega(c_k,t_k,o)
+
\lambda_c \nabla_c Q_c(o,c_k)
\right)
$$

其中：

```text
lambda_c = condition_guidance_scale
```

直观解释：

1. condition flow policy \(p_\omega(c|o)\) 提供离线行为分布约束，使 \(c\) 倾向于落在数据中常见的 condition 区域。
2. \(Q_c(o,c)\) 提供在线任务价值信号，把采样方向推向长期价值更高的 condition。
3. 得到的 \(c\) 再作为 skill flow prior 的条件，生成 \(z\)。

对应代码：

```text
reskill/models/flow_prior.py::compute_flow_z_guided
reskill/models/bc_flow.py::sample_z_guided_torch
reskill/train_reskill_agent_res.py::condition_guidance_enabled
```

当前第一版实现中，condition guidance 和 skill latent guidance 不同时开启。也就是 condition-flow 实验中通常设置：

```text
use_condition_flow = 1
use_grad = 0
condition_use_grad = 1
```

## 7. OOD 与支撑集约束的理解

使用 flow matching 的动机是给高层 condition 和 skill latent 加入数据分布约束：

$$
c \sim p_\omega(c|s),\quad z \sim p_\psi(z|s,c)
$$

这会让采样结果倾向于离线数据的经验支撑集，从而降低 OOD condition、OOD latent 和 OOD decoded action 的风险。

但这不是严格的硬保证。原因是：

1. flow matching 学的是连续分布和速度场，不是对数据支撑集的投影算子。
2. 在线状态 \(s\) 可能偏离离线状态分布，此时 \(p_\omega(c|s)\) 的输出也可能不可靠。
3. guidance 梯度 \(\nabla_c Q_c\) 或 \(\nabla_z Q_z\) 会把样本向高价值方向推，可能推出离线高密度区域。
4. decoder 只在离线 latent 和状态附近可靠，不能严格保证所有 decoded action 都在行为数据 manifold 上。

因此更准确的表述是：

```text
Flow matching provides a soft support-preserving inductive bias.
It biases c and z toward the empirical behavior distribution, reducing OOD risk,
but it does not strictly guarantee in-support samples.
```
