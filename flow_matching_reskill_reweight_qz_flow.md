# Flow Matching ReSkill：条件重加权与 Q_z 引导

## 目标

新的流程把问题拆成两部分：

1. 离线阶段使用条件重加权，让 condition flow 更容易覆盖低频但合理的 condition。
2. 在线阶段不再使用 Q_c 间接评价 condition，而是使用 Q_z 直接评价真正送入 decoder 的 skill latent。

整体在线链路为：

```text
s -> condition flow -> c -> skill prior -> z -> decoder/residual -> action
```

其中，条件重加权主要改善：

```text
s -> c
```

Q_z guidance 主要改善：

```text
(s, c) -> z
```

## 变量定义

对每个离线 chunk，记：

```text
s_i        当前状态
a0_i       chunk 第一步原始动作
c_i        condition，当前定义为 chunk 第一步原始动作
z_i        SkillVAE encoder 得到的 skill latent
R_i        chunk return
```

当前 condition 定义为：

$$
c_i = a_{0,i}
$$

condition flow 的建模对象是：

$$
p_\theta(c \mid s)
$$

也就是直接建模数据集里的第一步行为动作分布。这样在线采样到的 c 仍然落在数据动作的支撑集附近。

## 为什么行为策略只拟合 a0

长尾问题本质上来自数据集中动作出现频率不均衡。因此，辅助行为策略只拟合数据集里的原始第一步动作：

$$
\pi_\phi(a_0 \mid s)
$$

condition flow 的 target 也是 a0，因此行为策略、权重计算、condition flow、skill prior 的 condition 语义完全一致：

```text
c = a0
```

所以推荐流程是：

```text
1. 用 a0 训练辅助行为策略。
2. 用 a0 计算每个 chunk 的重加权权重。
3. 用这个权重训练 condition flow。
4. condition flow 的 target 直接使用 c = a0。
```

这样权重反映的是原始数据中的长尾动作频率，condition flow 学到的也是原始数据动作支撑集上的 condition。

## tanh Gaussian 行为策略

为了估计某个第一步动作在当前状态下是否低密度，先训练一个 tanh Gaussian 行为策略。策略先在 pre-tanh 空间采样：

$$
u
\sim
\mathcal N
\left(
\mu_\phi(s),
\operatorname{diag}(\sigma_\phi^2(s))
\right)
$$

再经过 tanh 得到归一化动作：

$$
y = \tanh(u)
$$

如果环境动作范围为：

$$
a_0 \in [a_{\min},a_{\max}]^{d_a}
$$

则真实动作由下面的仿射变换得到：

$$
a_0 = c_{\text{scale}}\odot y + b_{\text{shift}}
$$

其中：

$$
c_{\text{scale}}
=
\frac{a_{\max}-a_{\min}}{2}
$$

$$
b_{\text{shift}}
=
\frac{a_{\max}+a_{\min}}{2}
$$

如果动作已经归一化到：

$$
a_0 \in [-1,1]^{d_a}
$$

则：

$$
c_{\text{scale}}=\mathbf 1
$$

$$
b_{\text{shift}}=\mathbf 0
$$

## 数据动作到 pre-tanh 空间的反变换

对数据集中的第一步动作，先映射到 tanh 输出空间：

$$
y_i
=
\frac{a_{0,i}-b_{\text{shift}}}{c_{\text{scale}}}
$$

为了避免 atanh 在边界处数值发散，先进行裁剪：

$$
y_i
\leftarrow
\operatorname{clip}
\left(
y_i,
-1+\epsilon,
1-\epsilon
\right)
$$

然后反变换到 pre-tanh 空间：

$$
u_i
=
\operatorname{atanh}(y_i)
$$

其中：

$$
\operatorname{atanh}(y_i)
=
\frac{1}{2}
\log
\frac{1+y_i}{1-y_i}
$$

## tanh Gaussian log likelihood

行为策略在数据动作上的 log likelihood 为：

$$
\log \pi_\phi(a_{0,i}\mid s_i)
=
\log
\mathcal N
\left(
u_i;
\mu_\phi(s_i),
\operatorname{diag}(\sigma_\phi^2(s_i))
\right)
-
\sum_{k=1}^{d_a}
\log(1-y_{i,k}^2)
-
\sum_{k=1}^{d_a}
\log c_{\text{scale},k}
$$

其中 Gaussian 部分为：

$$
\log
\mathcal N
\left(
u_i;
\mu_i,
\operatorname{diag}(\sigma_i^2)
\right)
=
-
\frac{1}{2}
\sum_{k=1}^{d_a}
\left[
\left(
\frac{u_{i,k}-\mu_{i,k}}
{\sigma_{i,k}}
\right)^2
+
2\log\sigma_{i,k}
+
\log(2\pi)
\right]
$$

这里：

$$
\mu_i = \mu_\phi(s_i)
$$

$$
\sigma_i = \sigma_\phi(s_i)
$$

如果动作已经处于归一化范围，则尺度修正项为 0：

$$
\sum_{k=1}^{d_a}
\log c_{\text{scale},k}
=
0
$$

## 条件重加权

这一节只说明最终训练权重 w_i 怎么计算。输入是离线 chunk 的状态和第一步动作：

$$
(s_i,a_{0,i})
$$

输出是这个 chunk 的 condition flow loss 权重：

$$
w_i
$$

### 1. 计算 tanh Gaussian log probability

先用训练好的 tanh Gaussian 行为策略计算数据动作的 log probability：

$$
\ell_i
=
\log \pi_\phi(a_{0,i}\mid s_i)
$$

这里的 log probability 使用上一节的 tanh Gaussian 公式，需要包含两部分：

```text
1. pre-tanh Gaussian log probability
2. tanh 变换的 Jacobian 修正项
```

如果动作没有归一化到 [-1, 1]，还需要包含动作尺度变换的修正项。

### 2. 计算 raw log weight

用 log probability 构造反密度权重。先在 log 空间计算：

$$
\widetilde{\ell}^{w}_i
=
-
\beta
\ell_i
$$

也就是：

$$
\widetilde{\ell}^{w}_i
=
-
\beta
\log \pi_\phi(a_{0,i}\mid s_i)
$$

要求：

$$
0 \leq \beta < 1
$$

其中 beta 是重加权强度。beta 越大，低密度动作对应的权重越大；beta 等于 0 时，所有样本权重都退化为 1。

### 3. 全数据集归一化

为了保持 Flow Matching loss 的整体 scale 基本不变，在全数据集上计算 log mean weight：

$$
\log \bar w
=
\log
\left(
\frac{1}{N}
\sum_{j=1}^{N}
\exp(\widetilde{\ell}^{w}_j)
\right)
$$

然后得到归一化后的 log weight：

$$
\ell^{w,\text{norm}}_i
=
\widetilde{\ell}^{w}_i
-
\log \bar w
$$

再从 log 空间变回普通权重：

$$
w^{\text{norm}}_i
=
\exp
\left(
\ell^{w,\text{norm}}_i
\right)
$$

等价地：

$$
w^{\text{norm}}_i
=
\frac{
\exp
\left(
-
\beta
\log \pi_\phi(a_{0,i}\mid s_i)
\right)
}{
\frac{1}{N}
\sum_{j=1}^{N}
\exp
\left(
-
\beta
\log \pi_\phi(a_{0,j}\mid s_j)
\right)
}
$$

这样归一化后，全数据集上的平均权重接近 1：

$$
\frac{1}{N}
\sum_{i=1}^{N}
w^{\text{norm}}_i
\approx
1
$$

### 4. clip 得到最终权重

最后裁剪权重：

$$
w_i
=
\operatorname{clip}
\left(
w^{\text{norm}}_i,
w_{\min},
w_{\max}
\right)
$$

推荐初始设置：

```text
beta = 0.1 或 0.2
w_min = 0.2
w_max = 3.0
```

clip 的作用是避免极少数低似然样本权重过大，从而主导 condition flow 的训练。

### 5. 保存归一化统计

训练时不需要保存每条样本的 w_i，也不需要保存权重表。行为策略已经保存，当前 batch 里也有 s_i 和 a0_i，所以可以动态计算当前 batch 的权重。

离线预处理阶段只保存全数据集归一化常数：

$$
\log \bar w
$$

它保存在：

```text
condition_weight_stats.json
```

训练 condition flow 时，对当前 batch 的每个样本动态计算：

$$
\widetilde{\ell}^{w}_i
=
-
\beta
\log \pi_\phi(a_{0,i}\mid s_i)
$$

然后用保存的全局常数归一化：

$$
\ell^{w,\text{norm}}_i
=
\widetilde{\ell}^{w}_i
-
\log \bar w
$$

最终得到：

$$
w_i
=
\operatorname{clip}
\left(
\exp(\ell^{w,\text{norm}}_i),
w_{\min},
w_{\max}
\right)
$$

所以每次训练 condition flow 时的关系是：

$$
(s_i, a_{0,i})
\rightarrow
\pi_\phi(a_{0,i}\mid s_i)
\rightarrow
w_i
\rightarrow
(s_i, c_{\text{train},i})
$$

这样不依赖 dataset 的随机采样 index，也不会出现权重表和数据顺序错位的问题。

## 离线训练流程

### 1. 训练 SkillVAE

SkillVAE 学习 chunk 的 skill latent：

```text
encoder: chunk states/actions -> z
decoder: s, z -> action
```

标准训练目标为：

$$
\mathcal L_{\text{VAE}}
=
\mathcal L_{\text{BC}}
+
\lambda_{\text{KL}}\mathcal L_{\text{KL}}
$$

训练完成后保存 best checkpoint，并用 best SkillVAE 进入下一阶段。

### 2. 训练辅助行为策略

用离线数据中的第一步动作训练 tanh Gaussian 行为策略：

$$
\pi_\phi(a_0 \mid s)
$$

训练目标是最大化行为动作的 log likelihood，等价于最小化负对数似然：

$$
\mathcal L_{\text{beh}}
=
-
\mathbb E_{(s_i,a_{0,i})}
\left[
\log \pi_\phi(a_{0,i}\mid s_i)
\right]
$$

这里的 log likelihood 必须包含 tanh 变换的 Jacobian 修正项，而不是直接在动作空间用普通 Gaussian 密度。

训练好后，对每个 chunk 计算：

$$
w_i = w(s_i,a_{0,i})
$$

权重计算流程是：

```text
1. 将 a0 映射到 tanh 输出空间 y。
2. 对 y 做边界裁剪。
3. 用 atanh 得到 pre-tanh 动作 u。
4. 计算 tanh Gaussian log likelihood。
5. 用 -beta * log likelihood 得到 log-weight。
6. 在全数据集上归一化。
7. clip 到 [w_min, w_max]。
```

如果担心行为策略在训练集上过拟合，可以使用 cross-fitting：用其他 fold 训练行为策略，再给当前 fold 的样本计算权重。

### 3. 训练重加权 condition flow

condition flow 建模：

$$
p_\theta(c \mid s)
$$

训练 target 使用：

$$
c_i
=
a_{0,i}
$$

对一个 batch，数据为：

$$
\left\{
(s_i,c_i,w_i)
\right\}_{i=1}^{B}
$$

Flow Matching 训练时，先采样 base noise：

$$
x_{0,i}
\sim
p_0
$$

把真实 condition target 记为：

$$
x_{1,i}
=
c_i
$$

再采样时间：

$$
t_i
\sim
\operatorname{Uniform}(0,1)
$$

构造插值点：

$$
x_{t,i}
=
(1-t_i)x_{0,i}
+
t_i x_{1,i}
$$

对应的 target velocity 是：

$$
u_i
=
x_{1,i}
-
x_{0,i}
$$

condition flow 预测：

$$
\hat u_i
=
v_\theta(t_i,x_{t,i},s_i)
$$

先计算每个样本自己的 Flow Matching loss，不要先对 batch 求平均：

$$
\ell_i
=
\left\|
\hat u_i
-
u_i
\right\|^2
$$

然后用 w_i 加权：

$$
\mathcal L_c
=
\frac{
\sum_{i=1}^{B}
w_i\ell_i
}{
\sum_{i=1}^{B}
w_i
+
\epsilon
}
$$

如果实现里已经把 w_i 全数据集归一化到均值接近 1，也可以写成：

$$
\mathcal L_c
=
\frac{1}{B}
\sum_{i=1}^{B}
w_i\ell_i
$$

但更稳的是使用除以 batch 内权重和的形式，避免不同 batch 的权重均值波动影响 learning rate。

这里的关键点是：

```text
权重来自 a0。
condition flow 的训练 target 是 c = a0。
每个样本先算自己的 FM squared error。
w_i 只乘在这个样本的 squared error 上。
不要把权重乘到 flow target velocity 上。
不要把权重乘到 c、x_t 或 u_i 上。
```

### 4. 训练 skill prior

skill prior 建模：

$$
p_\psi(z \mid s,c)
$$

它负责在给定状态和 condition 后生成 skill latent。训练数据中的 z 来自 SkillVAE encoder：

$$
z_i = \operatorname{Encoder}(chunk_i)
$$

skill prior 的 Flow Matching 目标为：

$$
\mathcal L_z
=
\left\|
v_\psi(t,z_t,s_i,c_i)-u_t
\right\|^2
$$

skill prior 是否使用同一个权重，需要作为消融项。保守做法是先只对 condition flow 加权，因为当前想解决的是低频 condition 采不到的问题。

## 在线训练流程

### 在线采样

在线阶段，先从重加权训练后的 condition flow 采样：

$$
c_t \sim p_\theta(c \mid s_t)
$$

然后从 skill prior 采样：

$$
z_t \sim p_\psi(z \mid s_t,c_t)
$$

再由 decoder 产生离线 skill 动作：

$$
a_{\text{dec},t}
=
\operatorname{Decoder}(s_t,z_t)
$$

residual policy 输出 residual action：

$$
\tilde a_{\text{res},t}
=
\pi_{\text{res}}
\left(
s_t,
z_t,
a_{\text{dec},t}
\right)
$$

为了限制 residual 对离线 skill 的破坏，先将 residual action 裁剪到固定边界：

$$
a_{\text{res},t}
=
\operatorname{clip}
\left(
\tilde a_{\text{res},t},
-0.5,
0.5
\right)
$$

最终执行动作是：

$$
a_t
=
a_{\text{dec},t}
+
\alpha_t a_{\text{res},t}
$$

其中 alpha_t 是 residual factor，由 logistic schedule 给出，并且可以设置任务相关上限：

$$
\alpha_t
=
\min
\left(
\operatorname{logistic}(t),
\alpha_{\max}
\right)
$$

### 使用 Q_z，而不是 Q_c

新方案不再学习：

$$
Q_c(s,c)
$$

而是学习：

$$
Q_z(s,z)
$$

原因是实际执行的 chunk 更直接由 z 决定，而不是由 c 决定。c 只通过下面的随机映射影响动作：

$$
c \rightarrow z \rightarrow a
$$

因此，Q_c 是对 condition 的间接评价，而 Q_z 是对实际 skill latent 的直接评价。

每执行一个 chunk 后，向 replay buffer 存入：

$$
(s_t,z_t,R_t,s_{t+H},d_t)
$$

其中：

```text
H     chunk 长度
R_t   chunk return
d_t   done 标记
```

TD target 为：

$$
y_t
=
R_t
+
(1-d_t)
\gamma^H
Q_z^{\text{target}}(s_{t+H},z_{t+H})
$$

其中下一个 latent 通过当前策略采样：

$$
c_{t+H}
\sim
p_\theta(c\mid s_{t+H})
$$

$$
z_{t+H}
\sim
p_\psi(z\mid s_{t+H},c_{t+H})
$$

### Q_z guidance

在线采样时，先正常采样 condition：

$$
c_t \sim p_\theta(c\mid s_t)
$$

然后在采样 z_t 时使用 Q_z guidance：

$$
z_t
\sim
p_\psi(z\mid s_t,c_t)
\quad
\text{guided by}
\quad
\nabla_z Q_z(s_t,z)
$$

这样职责分工更清晰：

```text
condition reweighting:
    解决长尾 condition 采不到的问题。

Q_z guidance:
    在给定 condition 后，选择长期价值更高的 skill latent。

residual policy:
    修正 decoder 动作细节。
```

## 与 Q_c guidance 的区别

Q_c 版本学习：

$$
Q_c(s,c)
$$

并用它引导 condition flow 采样。

问题是 c 并不直接决定最终执行的动作。实际动作链路是：

$$
c
\rightarrow
z
\rightarrow
\operatorname{Decoder}(s,z)
\rightarrow
a
$$

同一个 c 下，z 仍然有采样随机性；decoder 也会进一步影响最终动作。因此，Q_c 的价值估计会包含从 c 到 z 再到 a 的随机性和噪声。

新方案使用：

$$
Q_z(s,z)
$$

直接评价送入 decoder 的 latent skill。这样可以减少多层随机映射带来的价值估计噪声。

## 建议消融实验

建议至少跑四组：

```text
A. 原始 condition flow，无 Q_z guidance
B. 重加权 condition flow，无 Q_z guidance
C. 原始 condition flow，加 Q_z guidance
D. 重加权 condition flow，加 Q_z guidance
```

解释方式：

```text
B > A:
    条件重加权改善了长尾 condition 覆盖。

C > A:
    Q_z guidance 本身有效。

D 最好:
    条件重加权和 Q_z guidance 互补。
```

建议同时在以下数据集上比较：

```text
fetch_block_40000
pick999:push1
pick1:push999
```

这样可以区分 full dataset 下的方法能力和长尾数据下的性能退化。

## 实现注意事项

### 1. 权重参数要保守

建议从：

```text
beta = 0.1 或 0.2
w_max = 3.0
```

开始。不要一开始使用很大的 beta 或 w_max，否则少数离群样本会主导训练。

### 2. 监控有效样本数

定义有效样本数：

$$
N_{\text{eff}}
=
\frac{
\left(\sum_i w_i\right)^2
}{
\sum_i w_i^2
}
$$

如果：

$$
\frac{N_{\text{eff}}}{N}
$$

太小，说明权重过于集中，训练会被少数样本支配。一般希望不要低于：

$$
0.1 \sim 0.2
$$

### 3. condition target 不加噪声

当前 condition 定义为：

$$
c_i = a_{0,i}
$$

训练 condition flow 时也直接使用这个 target，不再使用旧的缩放加噪声 condition 构造。

原因是当前方法希望 condition flow 采样出的 c 保持在数据动作支撑集附近。给 condition target 加噪声会把训练分布扩到离线数据之外，削弱 flow model 的 on-support 假设。

### 4. tanh Gaussian 可能不足以描述多峰行为

如果真实行为策略明显多峰，单个 tanh Gaussian 仍然可能把多个模式压成一个均值和方差。这时 log likelihood 不一定能准确反映真实稀疏度。

后续可以尝试：

```text
Mixture tanh Gaussian
辅助 flow density model
局部 kNN 条件密度估计
```

### 5. 离线 MSE 不一定是唯一指标

重加权后，普通 validation MSE 不一定下降，因为训练目标已经从原始经验分布变成了重加权分布：

$$
q(c\mid s)
\propto
w(s,a_0)p_{\text{data}}(c\mid s)
$$

因此更重要的指标是：

```text
offline base rollout reward/success
online long-tail reward/success
condition sample coverage
权重分布
有效样本数
```
