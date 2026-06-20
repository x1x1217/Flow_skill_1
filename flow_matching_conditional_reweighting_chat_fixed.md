# 连续动作不平衡数据下的 Flow Matching 条件重加权讨论记录

> 本文档整理了本轮聊天中关于：  
> 1. 不平衡连续状态-动作数据集；  
> 2. 用 Flow Matching 建模 $p(a\mid s)$；  
> 3. 如何平衡每个状态下的条件动作分布；  
> 4. 不使用聚类时的连续权重设计；  
> 5. 使用高斯行为策略 $\pi(a\mid s)$ 的 log likelihood 设定权重。  
>
> 说明：这里只整理可见对话内容，不包含任何隐藏推理过程。

---

## 1. 初始问题：是否可以从速度场 loss 加权角度平衡动作分布？

### 用户问题

现在我有一个不平衡数据集，包含连续空间下的状态和动作。现在我希望用流匹配模型建模 $p(a\mid s)$，但是希望能够建模分布能够比较平衡。我现在想从流匹配模型的速度场更新权重的角度来实现，即对数据集中多的动作给一个较低的权重，给稀疏的动作一个较大的权重以实现。这样是否合理？如果不合理，有其他实现方法吗？如果合理，我如何为连续空间下的动作设定该权重？

### 回复整理

你的想法**可以合理**，但要非常明确：

你不是在更好地学习原始的 $p_{\text{data}}(a\mid s)$，而是在学习一个**重加权后的条件分布** $q(a\mid s)$。如果你的目标是“生成时动作分布更均衡”，这种方法是成立的；如果你的目标是“忠实估计数据里的行为策略分布”，那这种加权会引入偏差。

Flow Matching/Conditional Flow Matching 本质上是通过回归概率路径上的速度场来训练连续归一化流；标准目标是对数据分布下采样到的端点做速度场回归。把每个样本的 loss 乘上权重，相当于改变了这些端点在训练目标中的出现概率。

---

### 1.1 加权 FM 实际在学什么？

设数据为 $(s_i,a_i)$，流从噪声 $a_0\sim p_0$ 运输到数据动作 $a_1=a_i$。标准 conditional FM loss 可以写成：

$$
\mathcal L_{\text{FM}}
=
\mathbb E_{(s,a)\sim p_{\text{data}},\, a_0,\,t}
\left[
\left\|
v_\theta(t,x_t,s)-u_t(a_0,a)
\right\|_2^2
\right],
$$
其中 $x_t$ 是从 $a_0$ 到 $a$ 的插值点，$u_t$ 是对应的目标速度。

你提出的做法相当于：

$$
\mathcal L_{\text{wFM}}
=
\frac{
\mathbb E_{(s,a)\sim p_{\text{data}},\, a_0,\,t}
\left[
w(s,a)
\left\|
v_\theta(t,x_t,s)-u_t(a_0,a)
\right\|_2^2
\right]
}{
\mathbb E_{(s,a)\sim p_{\text{data}}}[w(s,a)]
}.
$$
这对应的目标 joint distribution 是：

$$
q(s,a)
=
\frac{w(s,a)p_{\text{data}}(s,a)}
{\mathbb E_{p_{\text{data}}}[w(s,a)]}.
$$
于是条件分布变成：

$$
q(a\mid s)
=
\frac{
w(s,a)p_{\text{data}}(a\mid s)
}{
\int w(s,a')p_{\text{data}}(a'\mid s)\, da'
}.
$$
所以结论是：

$$
\boxed{
\text{加权是合理的，但你学到的是 } q(a\mid s)，不是原始 } p_{\text{data}}(a\mid s).
}
$$
一个很重要的实现细节：**不要把目标速度 $u_t$ 乘以权重**。正确做法是乘 loss：

$$
w_i\|v_\theta-u_i\|^2,
$$
而不是让目标变成：

$$
v_\theta \approx w_i u_i.
$$
后者会改变速度场的物理意义，通常是不对的。

---

### 1.2 最大风险：不要只按全局动作密度盲目加权

如果你只用边缘动作密度 $p(a)$ 设权重，比如：

$$
w(a)\propto \frac{1}{\hat p(a)},
$$
那么你实际得到的是：

$$
q(a\mid s)
\propto
\frac{p_{\text{data}}(a\mid s)}{\hat p(a)}.
$$
这会有几个问题。

第一，动作稀疏不一定意味着“应该多生成”。它可能只是某些状态很少出现，也可能是噪声、异常值、低质量行为。

第二，动作的稀有性往往和状态绑定。例如某个动作只在特殊状态下合理，如果你用全局 $p(a)$ 加权，会把这类状态也一起放大，甚至让模型在相近但不适合的状态下也更倾向于生成这些动作。

第三，如果动作空间维度较高，直接估计 $\hat p(a)$ 很不稳定。KDE、kNN 密度在高维下容易受尺度、噪声和局部样本不足影响。

所以更推荐的目标不是“让全局 $p(a)$ 变平”，而是先定义你想要的平衡方式：

$$
\text{你是想平衡 } p(a)，\quad
\text{还是想平衡每个状态下的 } p(a\mid s)?
$$
这两个目标完全不同。

---

### 1.3 合理的权重设计方式

#### 方法 A：按边缘动作密度加权，适合只关心全局动作覆盖

这是最直接的版本：

$$
\tilde w_i
=
\left(
\frac{\bar \rho}{\hat \rho_A(a_i)+\epsilon}
\right)^\beta,
$$
然后归一化：

$$
w_i
=
\frac{\tilde w_i}
{\frac{1}{N}\sum_{j=1}^N \tilde w_j}.
$$
其中 $\hat \rho_A(a_i)$ 是动作 $a_i$ 的连续密度估计，$\beta\in[0,1]$ 控制重平衡强度。

常用设置：

$$
\beta=0
$$
表示不加权；

$$
\beta=1
$$
近似完全反密度加权；

$$
\beta\in[0.25,0.7]
$$
通常更稳。

还应该 clip：

$$
w_i \leftarrow \text{clip}(w_i,w_{\min},w_{\max}).
$$
例如：

$$
w_{\min}=0.2,\quad w_{\max}=5 \text{ 或 } 10.
$$
---

#### 方法 B：按条件动作密度加权，通常更适合 $p(a\mid s)$

如果真的关心 conditional generator，更推荐：

$$
\tilde w_i
=
\left(
\frac{\bar \rho(s_i)}{\hat \rho(a_i\mid s_i)+\epsilon}
\right)^\beta.
$$
也就是：不是问“这个动作在全数据里罕见吗”，而是问：

$$
\boxed{
\text{在类似状态 } s_i \text{ 下，这个动作 } a_i \text{ 罕见吗？}
}
$$
这更符合建模 $p(a\mid s)$ 的目标。

一个非参数估计形式是：

$$
\hat \rho(a_i\mid s_i)
=
\frac{
\sum_j K_s\left(\frac{s_i-s_j}{h_s}\right)
K_a\left(\frac{a_i-a_j}{h_a}\right)
}{
\sum_j K_s\left(\frac{s_i-s_j}{h_s}\right)
}.
$$
直觉是：先找和 $s_i$ 相近的状态，再看 $a_i$ 在这些邻近状态对应的动作里是否稀有。

实际中，如果 $s$ 维度很高，不建议直接在原始状态空间做 KDE。可以用：

$$
z_s=f_\psi(s)
$$
作为状态 embedding，然后在 $z_s$ 上找近邻。这个 embedding 可以来自 dynamics model、policy encoder、contrastive representation、PCA、autoencoder，或者当前 FM 模型的 condition encoder。

---

#### 方法 C：状态簇内归一化，防止改变状态分布

如果用的是动作边缘权重 $w(a)$，最好再做一个状态局部归一化，否则高权重动作对应的状态也会被整体放大。

设把状态分成若干簇 $g(s)$，先算原始权重：

$$
\tilde w_i
=
\left(
\frac{1}{\hat \rho_A(a_i)+\epsilon}
\right)^\beta.
$$
然后在每个状态簇里归一化：

$$
w_i
=
\frac{\tilde w_i}
{
\frac{1}{|\mathcal I_{g_i}|}
\sum_{j\in \mathcal I_{g_i}}\tilde w_j
}.
$$
这样可以近似保证：

$$
\mathbb E[w\mid g(s)]\approx 1.
$$
直观地说：你是在每个状态区域内部平衡动作，而不是让某些罕见状态整体支配训练。

---

### 1.4 连续动作权重怎么估计？

#### 方案 1：KDE 密度估计

适合动作维度较低，例如 $d_a\leq 5$。

先标准化动作：

$$
a'_i = \frac{a_i-\mu_a}{\sigma_a}.
$$
然后做 KDE：

$$
\hat \rho_A(a_i)
=
\frac{1}{N}
\sum_{j=1}^N
\mathcal N(a'_i;a'_j,h^2I).
$$
权重：

$$
\tilde w_i
=
(\hat \rho_A(a_i)+\epsilon)^{-\beta}.
$$
然后：

$$
w_i
=
\text{clip}
\left(
\frac{\tilde w_i}{\text{mean}(\tilde w)},
w_{\min},
w_{\max}
\right).
$$
这种方法简单，但高维会变差。

---

#### 方案 2：kNN 密度估计

对每个动作 $a_i$，找第 $k$ 近邻距离 $r_{i,k}$。近似有：

$$
\hat \rho_A(a_i)
\propto
\frac{k}{N r_{i,k}^{d_a}}.
$$
所以反密度权重可以写成：

$$
\tilde w_i
\propto
r_{i,k}^{\beta d_a}.
$$
实际实现时更稳的是用 log：

$$
\log \tilde w_i
=
\beta d_a \log(r_{i,k}+\epsilon).
$$
然后减去均值、clip。

kNN 的好处是不需要选 KDE 带宽，但高维同样会受距离退化影响。

---

#### 方案 3：动作聚类后按簇频率加权

这是工程上最稳的版本之一。

先对动作做聚类：

$$
c_i = \text{Cluster}(a_i).
$$
例如 K-means、GMM、VQ-VAE codebook，或者在 action embedding 上聚类。

设第 $c$ 个动作簇有 $n_c$ 个样本，权重：

$$
\tilde w_i
=
\left(
\frac{N}{K n_{c_i}}
\right)^\beta.
$$
然后 normalize + clip。

这种方法牺牲了一些连续性，但很稳，尤其适合高维动作。

---

#### 方案 4：先训练一个条件密度模型，再用它给权重

这是更贴近目标的方法。

先训练一个粗略行为模型：

$$
r_\phi(a\mid s)\approx p_{\text{data}}(a\mid s).
$$
它可以是：

$$
\text{MDN},\quad \text{normalizing flow},\quad \text{energy model},\quad \text{VAE},\quad \text{另一个小型 FM}.
$$
然后对每个样本计算：

$$
\ell_i = \log r_\phi(a_i\mid s_i).
$$
权重设为：

$$
\tilde w_i
=
\exp(-\beta \ell_i).
$$
再做：

$$
w_i
=
\text{clip}
\left(
\frac{\tilde w_i}{\text{mean}(\tilde w)},
w_{\min},
w_{\max}
\right).
$$
这相当于：

$$
w_i \propto \frac{1}{\hat p(a_i\mid s_i)^\beta}.
$$
当 $\beta=1$ 时，接近条件反密度加权；当 $\beta<1$ 时，是更温和的 flattening。

---

### 1.5 推荐训练实现

FM loss 里按样本加权即可：

```python
# v_pred: [B, action_dim]
# u_t:    [B, action_dim]
# w:      [B], precomputed sample weights

per_sample_loss = ((v_pred - u_t) ** 2).sum(dim=-1)

w = w.detach()
w = torch.clamp(w, min=w_min, max=w_max)

loss = (w * per_sample_loss).sum() / (w.sum() + 1e-8)
```

不要写成：

```python
# 不推荐
loss = ((v_pred - w[:, None] * u_t) ** 2).sum(dim=-1).mean()
```

这会改变速度目标本身。

---

### 1.6 更稳的总体方案

建议按下面顺序做：

**第一版：动作聚类 + 状态簇内归一化。**

这通常比直接 KDE 更稳。

1. 标准化动作 $a$。
2. 对动作或 action embedding 聚类，得到 $c_i$。
3. 设：

$$
\tilde w_i
=
\left(
\frac{N}{K n_{c_i}}
\right)^\beta.
$$
4. 对状态也聚类，得到 $g_i$。
5. 在每个状态簇里归一化：

$$
w_i
=
\frac{\tilde w_i}
{
\text{mean}_{j:g_j=g_i}(\tilde w_j)
}.
$$
6. clip：

$$
w_i\leftarrow \text{clip}(w_i,0.2,10).
$$
7. 用 weighted FM loss 训练。

这个版本的优点是：稳定、简单、不会过度依赖高维连续密度估计。

---

### 1.7 什么时候不建议这样做？

如果目标是得到校准良好的行为模型：

$$
p_{\text{model}}(a\mid s)\approx p_{\text{data}}(a\mid s),
$$
那不要用这种权重作为最终训练目标。可以用它做预训练，然后再用原始 FM loss fine-tune：

$$
\mathcal L
=
\mathcal L_{\text{FM}}
$$
这样模型先关注 rare actions，再回到原始分布校准。

也可以用混合目标：

$$
\mathcal L
=
(1-\lambda)\mathcal L_{\text{FM}}
+
\lambda \mathcal L_{\text{wFM}}.
$$
其中 $\lambda\in[0,1]$。这比直接完全重加权更温和。

---

### 1.8 其他替代方法

除了 loss weighting，还有这些方法：

**重采样 endpoint。**  
按 $w_i$ 采样 $(s_i,a_i)$，然后用普通 FM loss。期望上和加权 loss 类似，但实现更简单。缺点是 rare sample 会被重复采样，可能过拟合。

**balanced mini-batch。**  
按动作簇或状态-动作簇构造 batch，让每个 batch 内 rare/common region 更均匀。这个通常比极端 sample weight 稳。

**两阶段训练。**  
先用 balanced objective 学 rare actions，再用原始数据分布 fine-tune，适合既想覆盖 rare action，又不想完全破坏概率校准。

**条件密度温度化。**  
如果能估计 $p(a\mid s)$，可以设目标：

$$
q(a\mid s)
\propto
p_{\text{data}}(a\mid s)^\tau,
\quad 0<\tau<1.
$$
$\tau$ 越小，分布越平。对应的样本权重近似为：

$$
w(s,a)
\propto
p_{\text{data}}(a\mid s)^{\tau-1}.
$$
这和反密度加权是一致的，其中：

$$
\beta = 1-\tau.
$$
---

### 1.9 第一部分结论

方法**合理，但不是在建模原始 $p(a\mid s)$**，而是在建模：

$$
q(a\mid s)
\propto
w(s,a)p_{\text{data}}(a\mid s).
$$
推荐的权重形式是：

$$
w_i
=
\text{clip}
\left(
\frac{
(\hat \rho(a_i\mid s_i)+\epsilon)^{-\beta}
}{
\text{mean}((\hat \rho(a_i\mid s_i)+\epsilon)^{-\beta})
},
w_{\min},
w_{\max}
\right),
$$
其中 $\beta\in[0.25,0.7]$ 通常比 $\beta=1$ 稳。

如果条件密度 $\hat\rho(a\mid s)$ 难估计，可以先用更工程稳健的版本：

$$
\boxed{
\text{动作聚类} \rightarrow \text{按簇频率反比加权} \rightarrow \text{状态簇内归一化} \rightarrow \text{clip} \rightarrow \text{weighted FM loss}
}
$$
---

## 2. 用户明确目标：平衡每个状态下的 $p(a\mid s)$

### 用户问题

我想平衡每个状态下的 $p(a\mid s)$。

### 回复整理

那你的权重应该设计成 **conditional weight**：

$$
w_i = w(s_i,a_i)
$$
而不是只依赖动作的全局权重 $w(a_i)$。

核心原则是：

$$
\boxed{
\mathbb E_{a\sim p_{\text{data}}(a\mid s)}[w(s,a)] \approx 1
}
$$
这样你只改变每个状态下的动作分布 $p(a\mid s)$，而不会顺便改变状态分布 $p(s)$。

---

### 2.1 真正要学的目标分布

原始数据分布是：

$$
p_{\text{data}}(s,a)=p_{\text{data}}(s)p_{\text{data}}(a\mid s)
$$
加权 FM loss 后，模型实际学习的是：

$$
q(s,a)\propto w(s,a)p_{\text{data}}(s,a)
$$
因此：

$$
q(a\mid s)
=
\frac{
w(s,a)p_{\text{data}}(a\mid s)
}{
\int w(s,a')p_{\text{data}}(a'\mid s)\,da'
}
$$
如果希望只平衡每个状态下的动作分布，而不改变状态分布，应该让：

$$
\int w(s,a')p_{\text{data}}(a'\mid s)\,da' = 1
$$
也就是：

$$
\boxed{
\mathbb E[w(s,a)\mid s]=1
}
$$
这样：

$$
q(s)=p_{\text{data}}(s)
$$
而：

$$
q(a\mid s)
$$
会变得比原来的 $p(a\mid s)$ 更平衡。

---

### 2.2 推荐理论形式：条件反密度加权

可以设：

$$
w(s,a)
=
\frac{
(\hat p(a\mid s)+\epsilon)^{-\beta}
}{
\mathbb E_{a'\sim p_{\text{data}}(a'\mid s)}
[(\hat p(a'\mid s)+\epsilon)^{-\beta}]
}
$$
其中：

$$
\beta\in[0,1]
$$
控制平衡强度。

当：

$$
\beta=0
$$
时：

$$
w(s,a)=1
$$
等价于不平衡处理，模型学习原始 $p(a\mid s)$。

当：

$$
0<\beta<1
$$
时，稀疏动作权重变大，高频动作权重变小，但不会过度极端。

当：

$$
\beta=1
$$
时，目标接近于把每个状态下的动作分布压平到类似 uniform over support，但这通常太激进，不推荐一开始就用。

加权后的条件分布近似为：

$$
q_\beta(a\mid s)
\propto
p_{\text{data}}(a\mid s)^{1-\beta}
$$
所以可以理解为：不是把 $p(a\mid s)$ 完全变均匀，而是在做 **conditional distribution tempering**。

更实用的参数是：

$$
\boxed{
\beta = 0.3 \sim 0.7
}
$$
建议先从：

$$
\beta=0.5
$$
开始。

---

### 2.3 连续状态下不能按“完全相同的 s”统计

因为状态 $s$ 是连续空间，几乎不会出现两个完全相同的状态，所以不能直接统计：

$$
\#(a\mid s)
$$
必须改成：

$$
\boxed{
\text{在 } s_i \text{ 的局部邻域内，估计 } a_i \text{ 的条件密度}
}
$$
也就是用相似状态集合：

$$
\mathcal N_i = \{j: s_j \approx s_i\}
$$
然后估计：

$$
\hat p(a_i\mid s_i)
$$
---

### 2.4 推荐实现：状态邻域内的动作簇频率加权

这是较稳的工程实现，比直接 KDE 稳得多。

#### Step 1：标准化状态和动作

先对状态和动作做标准化：

$$
\tilde s_i = \frac{s_i-\mu_s}{\sigma_s}
$$
$$
\tilde a_i = \frac{a_i-\mu_a}{\sigma_a}
$$
如果状态维度很高，不要直接在原始状态空间里找近邻。可以先学一个状态 embedding：

$$
z_i = f_\psi(s_i)
$$
然后在 $z_i$ 空间里找近邻。

#### Step 2：对动作做聚类

把连续动作离散成动作簇：

$$
c_i = C(a_i)
$$
例如用 K-means、GMM、VQ-VAE codebook 都可以。

假设共有 $K$ 个动作簇：

$$
c_i\in\{1,\dots,K\}
$$
#### Step 3：在每个状态附近统计动作簇分布

对每个样本 $i$，找到状态邻域：

$$
\mathcal N_i = \text{kNN}(s_i)
$$
然后统计邻域内动作簇频率：

$$
\hat p(c\mid s_i)
=
\frac{
\sum_{j\in\mathcal N_i}\mathbf 1[c_j=c]+\alpha
}{
|\mathcal N_i|+\alpha K
}
$$
这里 $\alpha$ 是平滑项，防止某些簇频率为 0。常用：

$$
\alpha=0.5,\ 1,\ 2
$$
#### Step 4：给样本 $i$ 设置 conditional weight

如果样本 $i$ 的动作簇是 $c_i$，那么：

$$
\tilde w_i
=
\hat p(c_i\mid s_i)^{-\beta}
$$
为了保证只改变 $p(a\mid s)$，不改变 $p(s)$，要做条件归一化：

$$
Z(s_i)
=
\sum_{c=1}^{K}
\hat p(c\mid s_i)^{1-\beta}
$$
最终权重：

$$
\boxed{
w_i
=
\frac{
\hat p(c_i\mid s_i)^{-\beta}
}{
\sum_{c=1}^{K}
\hat p(c\mid s_i)^{1-\beta}
}
}
$$
这样可以保证近似：

$$
\mathbb E[w_i\mid s_i]\approx 1
$$
也就是说，你是在每个状态附近平衡动作，而不是改变状态采样分布。

---

### 2.5 对应的目标条件分布

这个权重会把原来的动作簇分布：

$$
p(c\mid s)
$$
变成：

$$
q(c\mid s)
=
\frac{
p(c\mid s)^{1-\beta}
}{
\sum_{c'}p(c'\mid s)^{1-\beta}
}
$$
例如某个状态附近原本动作簇分布是：

$$
p(c\mid s) = [0.80,\ 0.15,\ 0.05]
$$
如果 $\beta=0.5$，则：

$$
q(c\mid s)
\propto
[\sqrt{0.80},\sqrt{0.15},\sqrt{0.05}]
$$
会变得更平衡，但不是完全均匀。

如果 $\beta=1$，则：

$$
q(c\mid s)
\propto
[1,1,1]
$$
相当于把所有动作簇变成均匀分布。这通常会过强，容易放大噪声动作或异常动作。

---

### 2.6 PyTorch 训练时怎么用

Flow Matching 的速度场目标还是原来的：

$$
u_t
$$
不要改成：

$$
w_i u_t
$$
正确做法是只给 loss 加权：

```python
# v_pred: [B, action_dim]
# u_t:    [B, action_dim]
# w:      [B], 每个样本预先计算好的 conditional weight

per_sample_loss = ((v_pred - u_t) ** 2).sum(dim=-1)

w = w.detach()
w = torch.clamp(w, min=0.2, max=10.0)

loss = (w * per_sample_loss).sum() / (w.sum() + 1e-8)
```

不要这样做：

```python
# 不推荐
loss = ((v_pred - w[:, None] * u_t) ** 2).sum(dim=-1).mean()
```

因为这会改变速度场目标本身，而不是改变样本在目标分布中的权重。

---

### 2.7 完整权重预计算伪代码

推荐版本：**状态 kNN + 动作聚类 + conditional inverse frequency**。

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

def compute_conditional_action_weights(
    states,
    actions,
    num_action_clusters=64,
    num_state_neighbors=512,
    beta=0.5,
    alpha=1.0,
    w_min=0.2,
    w_max=10.0,
):
    """
    states:  [N, state_dim]
    actions: [N, action_dim]

    return:
        weights: [N]
    """

    N = states.shape[0]

    # 1. 标准化
    state_scaler = StandardScaler()
    action_scaler = StandardScaler()

    S = state_scaler.fit_transform(states)
    A = action_scaler.fit_transform(actions)

    # 2. 动作聚类
    kmeans = MiniBatchKMeans(
        n_clusters=num_action_clusters,
        batch_size=4096,
        random_state=0,
        n_init="auto",
    )
    action_cluster = kmeans.fit_predict(A)

    # 3. 状态空间 kNN
    knn = NearestNeighbors(
        n_neighbors=num_state_neighbors,
        algorithm="auto",
    )
    knn.fit(S)
    neighbor_idx = knn.kneighbors(S, return_distance=False)

    # 4. 计算 conditional weights
    weights = np.zeros(N, dtype=np.float64)

    K = num_action_clusters

    for i in range(N):
        neigh = neighbor_idx[i]
        neigh_clusters = action_cluster[neigh]

        counts = np.bincount(neigh_clusters, minlength=K).astype(np.float64)

        # p(c | s_i)
        p_c_given_s = (counts + alpha) / (num_state_neighbors + alpha * K)

        c_i = action_cluster[i]

        # Z(s_i) = sum_c p(c|s_i)^(1-beta)
        Z = np.sum(p_c_given_s ** (1.0 - beta))

        weights[i] = (p_c_given_s[c_i] ** (-beta)) / Z

    # 5. 全局均值归一化，保持整体 loss scale 稳定
    weights = weights / (weights.mean() + 1e-8)

    # 6. clip，防止 rare/noisy action 权重爆炸
    weights = np.clip(weights, w_min, w_max)

    # 7. clip 后再归一化一次
    weights = weights / (weights.mean() + 1e-8)

    return weights
```

然后 dataset 返回：

```python
return s_i, a_i, weight_i
```

训练 FM 时用：

```python
loss = (weight_i * per_sample_fm_loss).sum() / (weight_i.sum() + 1e-8)
```

---

### 2.8 如果动作维度很低，可以用条件 KDE

如果动作维度 $d_a$ 比较低，例如：

$$
d_a \leq 4 \text{ 或 } 5
$$
也可以直接估计：

$$
\hat p(a_i\mid s_i)
$$
一种形式是：

$$
\hat p(a_i\mid s_i)
=
\frac{
\sum_{j=1}^{N}
K_s(s_i,s_j)K_a(a_i,a_j)
}{
\sum_{j=1}^{N}K_s(s_i,s_j)
}
$$
其中：

$$
K_s(s_i,s_j)
=
\exp\left(
-\frac{\|z_i-z_j\|^2}{2h_s^2}
\right)
$$
$$
K_a(a_i,a_j)
=
\exp\left(
-\frac{\|a_i-a_j\|^2}{2h_a^2}
\right)
$$
然后：

$$
\tilde w_i
=
(\hat p(a_i\mid s_i)+\epsilon)^{-\beta}
$$
再做局部归一化：

$$
w_i
=
\frac{
\tilde w_i
}{
\frac{1}{|\mathcal N_i|}
\sum_{j\in\mathcal N_i}\tilde w_j
}
$$
最后 clip：

$$
w_i\leftarrow \text{clip}(w_i,w_{\min},w_{\max})
$$
不过优先用“动作聚类版本”，因为 KDE 在高维动作空间里很容易不稳定。

---

### 2.9 参数怎么选

比较稳的初始设置是：

$$
\beta=0.5
$$
$$
K_{\text{action}}=32 \sim 128
$$
$$
K_{\text{state-neighbor}}=256 \sim 2048
$$
$$
\alpha=1.0
$$
$$
w_{\min}=0.2,\quad w_{\max}=5 \text{ 或 } 10
$$
如果发现模型仍然偏向多数动作，可以增大：

$$
\beta: 0.5 \rightarrow 0.7
$$
如果发现模型开始生成奇怪动作、噪声动作、低质量动作，就减小：

$$
\beta: 0.5 \rightarrow 0.3
$$
或者降低：

$$
w_{\max}: 10 \rightarrow 5 \rightarrow 3
$$
---

### 2.10 一个更稳的改进：只平衡动作模式，不平衡异常点

连续动作下最危险的是：某些动作稀疏是因为它们是异常点，而不是有价值的少数模式。

所以不建议直接给每个孤立动作极高权重。更稳的做法是：

$$
\boxed{
\text{先把动作聚成模式，再平衡模式频率}
}
$$
也就是说，平衡的是：

$$
p(c\mid s)
$$
而不是直接把每一个连续动作点都当成一个类别去平衡。

这样模型会更关注稀有动作模式，而不是孤立噪声样本。

---

### 2.11 第二部分结论

目标是平衡每个状态下的 $p(a\mid s)$，建议用：

$$
\boxed{
w_i
=
\frac{
\hat p(c_i\mid s_i)^{-\beta}
}{
\sum_{c=1}^{K}
\hat p(c\mid s_i)^{1-\beta}
}
}
$$
其中：

$$
c_i = \text{action cluster}(a_i)
$$
$$
\hat p(c\mid s_i)
=
\frac{
\sum_{j\in\mathcal N_i}\mathbf 1[c_j=c]+\alpha
}{
|\mathcal N_i|+\alpha K
}
$$
然后用 weighted FM loss：

$$
\mathcal L
=
\frac{
\sum_i w_i
\left\|
v_\theta(t,x_t,s_i)-u_t
\right\|^2
}{
\sum_i w_i
}
$$
这个方法的意义是：

$$
\boxed{
\text{在每个状态邻域内，降低高频动作模式的梯度贡献，提高低频动作模式的梯度贡献。}
}
$$
它比全局 $w(a)$ 更符合目标，也比直接连续 KDE 更稳。

---

## 3. 用户进一步要求：不想用聚类，有没有其他方法？

### 用户问题

聚类不太好，有其他合适方法吗？

### 回复整理

可以，不用聚类也有更合适的方法。既然目标是平衡每个状态下的 $p(a\mid s)$，更推荐下面三个方案，按实用性排序：

$$
\boxed{
\text{局部 kNN 稀疏度权重} \;>\; \text{条件 KDE 权重} \;>\; \text{辅助条件密度模型权重}
}
$$
聚类的问题是会把连续动作空间硬切成离散块，簇边界附近的样本权重可能不连续，而且簇数 $K$ 很敏感。连续目标不平衡问题里，更自然的做法是利用邻近目标之间的连续相似性，而不是把每个动作区域当成硬类别。

---

### 3.1 方案 1：局部状态邻域 + 动作 kNN 稀疏度权重

这是最推荐的无聚类方案。

对每个样本 $(s_i,a_i)$，先找状态邻域：

$$
\mathcal N_s(i)=\{j: s_j \text{ 和 } s_i \text{ 接近}\}
$$
然后在这些相似状态对应的动作集合里，看 $a_i$ 是否稀疏。

具体做法是：在 $\{a_j:j\in \mathcal N_s(i)\}$ 中，计算 $a_i$ 到第 $m$ 个最近动作的距离：

$$
r_i
=
\operatorname{kNNdist}_m
\left(
a_i,\{a_j:j\in \mathcal N_s(i)\}
\right)
$$
如果 $r_i$ 大，说明在类似状态下，动作 $a_i$ 周围样本少，是局部稀疏动作；如果 $r_i$ 小，说明动作 $a_i$ 在该状态附近很常见。

然后设：

$$
\tilde w_i = (r_i+\epsilon)^\gamma
$$
其中 $\gamma>0$ 控制平衡强度。最后做局部归一化：

$$
w_i
=
\frac{
\tilde w_i
}{
\frac{1}{|\mathcal N_s(i)|}
\sum_{j\in \mathcal N_s(i)} \tilde w_j
}
$$
这样可以近似保证：

$$
\mathbb E[w(s,a)\mid s]\approx 1
$$
也就是：只在每个状态附近平衡动作分布，而不是改变状态分布。

推荐初始参数：

$$
|\mathcal N_s(i)| = 256 \sim 2048
$$
$$
m = 8 \sim 32
$$
$$
\gamma = 0.5 \sim 2.0
$$
$$
w_{\min}=0.2,\quad w_{\max}=5 \text{ 或 } 10
$$
这个方法比动作聚类更连续，也比 KDE 更少依赖带宽。

---

### 3.2 方案 1 的实现代码

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def compute_local_action_knn_weights(
    states,
    actions,
    k_state=512,
    k_action=16,
    gamma=1.0,
    w_min=0.2,
    w_max=10.0,
    eps=1e-8,
):
    """
    states:  [N, state_dim]
    actions: [N, action_dim]

    返回:
        weights: [N]

    思路:
        对每个样本 i:
        1. 找 s_i 附近的 k_state 个状态邻居
        2. 在这些邻居的动作里，计算 a_i 的第 k_action 近邻距离 r_i
        3. r_i 越大，说明 a_i 在该状态附近越稀疏，权重越大
        4. 对状态邻域做归一化，避免改变 p(s)
    """

    N = states.shape[0]

    state_scaler = StandardScaler()
    action_scaler = StandardScaler()

    S = state_scaler.fit_transform(states)
    A = action_scaler.fit_transform(actions)

    # 多取 1 个，因为最近邻通常是自己
    nn_state = NearestNeighbors(n_neighbors=min(k_state + 1, N), algorithm="auto")
    nn_state.fit(S)
    state_neighbors = nn_state.kneighbors(S, return_distance=False)

    # 去掉自己
    state_neighbors = np.array([
        neigh[neigh != i][:k_state]
        for i, neigh in enumerate(state_neighbors)
    ])

    log_sparse = np.zeros(N, dtype=np.float64)

    for i in range(N):
        neigh = state_neighbors[i]
        A_neigh = A[neigh]

        # 在相似状态对应的动作中，计算 a_i 的动作距离
        dists = np.linalg.norm(A_neigh - A[i], axis=1)

        kk = min(k_action, len(dists))
        r_i = np.partition(dists, kk - 1)[kk - 1]

        # r_i 大 => 局部动作稀疏 => 权重大
        log_sparse[i] = np.log(r_i + eps)

    # raw weight
    # 减去中位数只是为了数值稳定
    raw = np.exp(gamma * (log_sparse - np.median(log_sparse)))

    # 局部状态归一化：近似保持 E[w | s] = 1
    weights = np.zeros(N, dtype=np.float64)

    for i in range(N):
        neigh = state_neighbors[i]
        local_mean = raw[neigh].mean()
        weights[i] = raw[i] / (local_mean + eps)

    # clip，防止异常点权重爆炸
    weights = np.clip(weights, w_min, w_max)

    # 全局均值归一化，保持 loss scale 稳定
    weights = weights / (weights.mean() + eps)

    return weights
```

FM 训练时仍然只加权 loss：

```python
per_sample_loss = ((v_pred - u_t) ** 2).sum(dim=-1)

w = weight_batch.detach()
loss = (w * per_sample_loss).sum() / (w.sum() + 1e-8)
```

不要把目标速度改成 $w_i u_t$。权重应该改变样本对回归目标的贡献，而不是改变速度目标的物理含义。

---

### 3.3 方案 2：局部条件 KDE 权重

比 kNN 稀疏度更“密度化”的版本是估计：

$$
\hat p(a_i\mid s_i)
$$
但不是全局 KDE，而是 **状态局部 KDE**。

先找状态邻域：

$$
\mathcal N_s(i)
$$
然后只用这些邻居的动作估计 $a_i$ 的条件密度：

$$
\hat p(a_i\mid s_i)
=
\frac{1}{|\mathcal N_s(i)|}
\sum_{j\in \mathcal N_s(i)}
K_a(a_i,a_j)
$$
例如：

$$
K_a(a_i,a_j)
=
\exp
\left(
-\frac{\|a_i-a_j\|^2}{2h_a^2}
\right)
$$
然后设：

$$
\tilde w_i
=
(\hat p(a_i\mid s_i)+\epsilon)^{-\beta}
$$
再做状态局部归一化：

$$
w_i
=
\frac{
\tilde w_i
}{
\frac{1}{|\mathcal N_s(i)|}
\sum_{j\in \mathcal N_s(i)} \tilde w_j
}
$$
这里：

$$
\beta\in[0,1]
$$
控制平衡强度。推荐先用：

$$
\beta=0.3 \sim 0.6
$$
不要一开始用 $\beta=1$，因为那会过度放大局部极低密度动作，容易把异常点也学进去。

这个方法的目标分布近似是：

$$
q(a\mid s)
\propto
p_{\text{data}}(a\mid s)^{1-\beta}
$$
所以它不是强行变成均匀分布，而是把原来的条件分布做温度化：

$$
p(a\mid s)
\quad\longrightarrow\quad
p(a\mid s)^{1-\beta}
$$
$\beta$ 越大，越平衡；$\beta$ 越小，越接近原始数据分布。

---

### 3.4 方案 3：训练一个辅助条件密度模型，再用 NLL 设权重

如果状态和动作维度都比较高，手工 kNN/KDE 可能不够稳。可以先训练一个辅助模型：

$$
r_\phi(a\mid s)\approx p_{\text{data}}(a\mid s)
$$
然后对每个训练样本计算：

$$
\ell_i = -\log r_\phi(a_i\mid s_i)
$$
$\ell_i$ 越大，说明该动作在该状态下越稀疏。权重设为：

$$
\tilde w_i
=
\exp(\beta \ell_i)
$$
等价于：

$$
\tilde w_i
=
r_\phi(a_i\mid s_i)^{-\beta}
$$
然后局部或条件归一化：

$$
w_i
=
\frac{
\tilde w_i
}{
\text{local-mean}_{j\in\mathcal N_s(i)}(\tilde w_j)
}
$$
适合做 $r_\phi(a\mid s)$ 的模型包括：

$$
\text{MDN},\quad
\text{conditional normalizing flow},\quad
\text{VAE},\quad
\text{autoregressive density model}
$$
这个方案有一个关键实现细节：**最好用 cross-fitting**。

也就是把数据分成 $K$ 折，辅助密度模型在 $K-1$ 折上训练，在剩下一折上给样本打分。否则辅助模型可能记住训练点，把训练样本密度估得过高，导致权重不可靠。

---

### 3.5 方案 4：直接学习权重，而不是估计 $p(a\mid s)$

还有一种方法是 density-ratio estimation。你不直接估计：

$$
p_{\text{data}}(a\mid s)
$$
而是直接估计：

$$
w(s,a)
=
\frac{q(a\mid s)}{p_{\text{data}}(a\mid s)}
$$
其中 $q(a\mid s)$ 是你想要的“更平衡”的目标条件分布。

例如，可以构造一个局部平滑的参考分布：

$$
q_{\text{ref}}(a\mid s_i)
=
\frac{1}{|\mathcal N_s(i)|}
\sum_{j\in\mathcal N_s(i)}
\mathcal N(a; a_j, \sigma_{\text{ref}}^2 I)
$$
这里 $\sigma_{\text{ref}}$ 比数据本身的局部带宽稍大，让 $q_{\text{ref}}$ 比原始条件分布更平滑、更均衡。

然后训练一个二分类器区分：

$$
(s,a)\sim p_{\text{data}}(s,a)
$$
和：

$$
(s,\tilde a)\sim p_{\text{data}}(s)q_{\text{ref}}(a\mid s)
$$
如果分类器输出：

$$
D(s,a)=P(y=1\mid s,a)
$$
并且 $y=1$ 表示来自 $q_{\text{ref}}$，那么密度比可以近似为：

$$
\frac{q_{\text{ref}}(a\mid s)}
{p_{\text{data}}(a\mid s)}
\approx
\frac{D(s,a)}{1-D(s,a)}
$$
这个比值就可以作为 FM loss 的样本权重。

这个方法的优点是灵活，缺点是必须明确设计 $q_{\text{ref}}(a\mid s)$。如果 $q_{\text{ref}}$ 设计得太宽，模型会被推向数据支撑外的动作。

---

### 3.6 第三部分建议

在“不想聚类”的前提下，优先用：

$$
\boxed{
\text{状态 kNN 邻域} + \text{动作 kNN 稀疏度} + \text{局部归一化}
}
$$
也就是方案 1。

它的权重形式很简单：

$$
r_i
=
\operatorname{kNNdist}_m
\left(
a_i,\{a_j:j\in \mathcal N_s(i)\}
\right)
$$
$$
\tilde w_i=(r_i+\epsilon)^\gamma
$$
$$
w_i
=
\frac{
\tilde w_i
}{
\frac{1}{|\mathcal N_s(i)|}
\sum_{j\in \mathcal N_s(i)}\tilde w_j
}
$$
$$
w_i \leftarrow \operatorname{clip}(w_i,w_{\min},w_{\max})
$$
它不需要动作聚类，不需要显式估计高维连续密度，也能表达“在类似状态下，哪些动作更稀疏”。

---

### 3.7 参数建议

比较稳的起点：

$$
k_{\text{state}}=512
$$
$$
k_{\text{action}}=16
$$
$$
\gamma=1.0
$$
$$
w_{\min}=0.2
$$
$$
w_{\max}=5.0
$$
如果模型仍然偏向多数动作：

$$
\gamma: 1.0 \rightarrow 1.5 \rightarrow 2.0
$$
如果模型开始生成奇怪动作或明显偏离数据支撑：

$$
\gamma: 1.0 \rightarrow 0.5
$$
或者：

$$
w_{\max}: 5 \rightarrow 3
$$
---

### 3.8 重要注意点

连续动作空间里，极端稀疏点可能是有意义的少数动作，也可能只是噪声或异常点。所以不建议让权重无限大。一定要做：

$$
\operatorname{clip}(w_i,w_{\min},w_{\max})
$$
并且最好监控加权后的有效样本数：

$$
N_{\text{eff}}
=
\frac{
(\sum_i w_i)^2
}{
\sum_i w_i^2
}
$$
如果：

$$
N_{\text{eff}} \ll N
$$
说明权重太集中，训练会被少量样本支配。一般希望：

$$
N_{\text{eff}}/N \geq 0.2
$$
至少不要低到 $0.05$ 以下。

最终训练目标还是：

$$
\mathcal L
=
\frac{
\sum_i w_i
\left\|
v_\theta(t,x_t,s_i)-u_t
\right\|^2
}{
\sum_i w_i
}
$$
也就是：**加权 loss，不加权速度目标。**

---

## 4. 用户提出：用高斯策略拟合行为策略，再用 log likelihood 设权重是否合理？

### 用户问题

现在我使用一个高斯策略 $\pi$ 拟合行为策略，然后计算每个动作在的似然值，即 $\log \pi(a\mid s)$，从而知道动作概率，然后在流匹配模型学习的时候修改权重，这样是否合理？

### 回复整理

合理，而且这是一个比“动作聚类”更自然的方案：你在做 **model-based conditional density reweighting**。

但要注意：连续动作下的 $\log \pi(a\mid s)$ 是**对条件密度的估计**，不是离散意义上的“概率”。它可以用来设计权重，但不能直接无脑用：

$$
w_i=\frac{1}{\pi(a_i\mid s_i)}
$$
否则很容易权重爆炸、改变状态分布，或者把高斯策略建模误差当成“稀有动作”。

---

### 4.1 这个方法在理论上对应什么？

FM 训练目标大概是：

$$
\mathcal L_{\text{FM}}
=
\mathbb E_{(s,a)\sim p_{\text{data}},t,a_0}
\left[
\left\|
v_\theta(t,x_t,s)-u_t
\right\|^2
\right]
$$
如果加样本权重：

$$
\mathcal L_{\text{wFM}}
=
\mathbb E
\left[
w(s,a)
\left\|
v_\theta(t,x_t,s)-u_t
\right\|^2
\right]
$$
那么模型实际学习的不是原始的：

$$
p_{\text{data}}(a\mid s)
$$
而是：

$$
q(a\mid s)
\propto
w(s,a)p_{\text{data}}(a\mid s)
$$
如果高斯策略满足：

$$
\pi_\phi(a\mid s)\approx p_{\text{data}}(a\mid s)
$$
然后取：

$$
w(s,a)
\propto
\pi_\phi(a\mid s)^{-\beta}
$$
那么加权后的目标分布近似为：

$$
q(a\mid s)
\propto
p_{\text{data}}(a\mid s)^{1-\beta}
$$
其中：

$$
\beta\in[0,1)
$$
控制平衡强度。

当 $\beta=0$ 时，不做平衡；

当 $\beta$ 越大时，条件动作分布越平；

当 $\beta\to 1$ 时，会非常激进，容易放大噪声和异常动作。

所以这个思路是合理的。它和连续不平衡回归里“根据目标值密度给稀有样本更大 loss 权重”的思路是一致的。

---

### 4.2 不要直接用原始的 $-\log \pi(a\mid s)$

最直接的想法是：

$$
\tilde w_i
=
\exp(-\beta \log \pi_\phi(a_i\mid s_i))
$$
也就是：

$$
\tilde w_i
=
\pi_\phi(a_i\mid s_i)^{-\beta}
$$
这个方向对，但还不够。你需要做**条件归一化**：

$$
w_i
=
\frac{
\pi_\phi(a_i\mid s_i)^{-\beta}
}{
Z(s_i)
}
$$
其中：

$$
Z(s_i)
=
\mathbb E_{a\sim p_{\text{data}}(a\mid s_i)}
[
\pi_\phi(a\mid s_i)^{-\beta}
]
$$
这样才能近似保证：

$$
\mathbb E[w(s,a)\mid s]\approx 1
$$
也就是说，只改变每个状态下的动作分布 $p(a\mid s)$，而不会额外改变状态分布 $p(s)$。

如果不做这个归一化，那么有些状态因为 $\pi_\phi(\cdot\mid s)$ 方差大、密度整体低，会被整体加大权重。这就不是“平衡每个状态下的动作”，而是在改变状态采样分布。

---

### 4.3 高斯策略下更好的权重公式

假设行为策略是：

$$
\pi_\phi(a\mid s)
=
\mathcal N(a;\mu_\phi(s),\Sigma_\phi(s))
$$
对样本 $(s_i,a_i)$，定义 Mahalanobis 距离：

$$
r_i^2
=
(a_i-\mu_\phi(s_i))^\top
\Sigma_\phi(s_i)^{-1}
(a_i-\mu_\phi(s_i))
$$
如果是 diagonal Gaussian：

$$
r_i^2
=
\sum_k
\left(
\frac{a_{i,k}-\mu_{\phi,k}(s_i)}
{\sigma_{\phi,k}(s_i)}
\right)^2
$$
那么推荐用这个权重：

$$
\boxed{
w_i
=
(1-\beta)^{d_a/2}
\exp\left(
\frac{\beta}{2}r_i^2
\right)
}
$$
其中 $d_a$ 是动作维度，且必须满足：

$$
0\leq \beta < 1
$$
这个公式的好处是：它已经做了高斯假设下的条件归一化。

注意这个公式里没有 $|\Sigma_\phi(s)|$ 的项。  
这很重要。原始的 $-\log \pi(a\mid s)$ 里面包含：

$$
\frac{1}{2}\log |\Sigma_\phi(s)|
$$
如果直接用 $\exp(-\beta\log\pi)$，状态方差大的地方会整体获得更大的权重，从而改变 $p(s)$。用上面的 Mahalanobis 形式可以避免这个问题。

---

### 4.4 实际实现建议

可以这样做：

```python
import math
import torch

def gaussian_conditional_balance_weight(
    actions,
    mu,
    log_std,
    beta=0.3,
    w_min=0.2,
    w_max=5.0,
    eps=1e-8,
):
    """
    actions: [B, action_dim]
    mu:      [B, action_dim]
    log_std: [B, action_dim]

    返回:
        w: [B]
    """

    assert 0.0 <= beta < 1.0

    action_dim = actions.shape[-1]

    # 防止 std 过小导致权重爆炸
    log_std = torch.clamp(log_std, min=-5.0, max=2.0)
    std = torch.exp(log_std) + eps

    z = (actions - mu) / std
    r2 = (z ** 2).sum(dim=-1)

    # log w = beta/2 * r^2 + d/2 * log(1-beta)
    log_w = 0.5 * beta * r2 + 0.5 * action_dim * math.log(1.0 - beta)

    # clip
    log_w = torch.clamp(
        log_w,
        min=math.log(w_min),
        max=math.log(w_max),
    )

    w = torch.exp(log_w)

    # batch 内再归一化，保持 loss scale 稳定
    w = w / (w.mean() + eps)

    return w.detach()
```

FM loss 用法：

```python
per_sample_loss = ((v_pred - u_t) ** 2).sum(dim=-1)

w = gaussian_conditional_balance_weight(
    actions=a_batch,
    mu=mu_batch,
    log_std=log_std_batch,
    beta=0.3,
)

loss = (w * per_sample_loss).sum() / (w.sum() + 1e-8)
```

不要这样做：

```python
# 不推荐
loss = ((v_pred - w[:, None] * u_t) ** 2).sum(dim=-1).mean()
```

权重应该乘在 loss 上，而不是乘在速度目标 $u_t$ 上。

---

### 4.5 如果坚持用 log pi，也应该这样写

如果策略不是普通 Gaussian，例如是 tanh-squashed Gaussian，或者希望保留完整 log prob，可以用更通用的形式：

$$
\tilde w_i
=
\exp(-\beta \log \pi_\phi(a_i\mid s_i))
$$
但需要估计局部归一化项：

$$
Z(s_i)
\approx
\frac{1}{|\mathcal N_s(i)|}
\sum_{j\in\mathcal N_s(i)}
\exp
\left(
-\beta
\log \pi_\phi(a_j\mid s_i)
\right)
$$
然后：

$$
w_i
=
\frac{
\exp(-\beta \log \pi_\phi(a_i\mid s_i))
}{
Z(s_i)+\epsilon
}
$$
这里 $\mathcal N_s(i)$ 是和 $s_i$ 相近的一批状态邻居，对应的动作 $a_j$ 可以近似看成来自 $p(a\mid s_i)$ 的局部样本。

实现上可以简化为：

$$
Z(s_i)
\approx
\text{local mean of } \tilde w
$$
也就是：

```python
raw_w_i = exp(-beta * log_pi_i)
w_i = raw_w_i / local_mean_raw_w_around_state_i
```

最后仍然需要：

```python
w_i = clip(w_i, w_min, w_max)
w_i = w_i / mean(w_i)
```

---

### 4.6 什么时候这个方法很好用？

这个方案在下面情况比较合适：

$$
p(a\mid s)
$$
在每个状态附近大致是单峰、椭圆形、连续变化的。

例如：

$$
a\mid s \approx \mathcal N(\mu(s),\Sigma(s))
$$
或者行为动作主要围绕一个主模式波动，只是某些偏离均值的动作比较少。

这时高斯策略的 Mahalanobis 距离：

$$
r_i^2
$$
确实可以表示“这个动作在当前状态下有多稀有”。

---

### 4.7 什么时候不太可靠？

最大问题是：**高斯策略可能不是一个好的 $p(a\mid s)$ 估计器。**

如果真实行为策略是多峰的，例如同一个状态下有两种合理动作模式：

$$
p(a\mid s)
=
0.7p_1(a\mid s)+0.3p_2(a\mid s)
$$
单个 Gaussian 会把这些模式揉成一个均值和方差。这样 $\log\pi(a\mid s)$ 反映的是“离高斯均值远不远”，不一定反映“在真实数据分布里稀不稀有”。

这会导致两个问题：

第一，少数模式可能被正确上权重，也可能只是因为离均值远而被过度放大。

第二，如果高斯方差很大，少数模式和多数模式的 log likelihood 差异可能反而被抹平。

所以如果 $p(a\mid s)$ 明显多峰，更建议把辅助行为策略换成：

$$
\text{Mixture Density Network}
$$
$$
\text{conditional normalizing flow}
$$
$$
\text{conditional VAE}
$$
$$
\text{小型 flow matching / diffusion behavior model}
$$
然后仍然用：

$$
w_i
=
\frac{
\hat p_\phi(a_i\mid s_i)^{-\beta}
}{
Z(s_i)
}
$$
这个框架不用变，只是把单高斯密度换成更可靠的条件密度估计。

---

### 4.8 关键细节

**第一，连续密度不是概率。**  
$\pi(a\mid s)$ 是 density，数值会受动作缩放影响。例如把动作单位从米变成厘米，density 数值会变。因此最好先对动作做标准化，或者在真实 action transform 下正确计算 log prob。

**第二，如果动作有界，要处理 tanh correction。**  
如果动作在 $[-1,1]$，并且策略是 tanh-squashed Gaussian，那么 $\log\pi(a\mid s)$ 需要包含 tanh 的 log-Jacobian correction。否则靠近边界的动作 likelihood 会估错。

**第三，最好不要用训练集 in-sample log likelihood。**  
如果策略网络比较强，它可能对训练样本过拟合，使得 log likelihood 不再可靠。更稳的是 cross-fitting：把数据分成几折，用其他折训练 $\pi_\phi$，再给当前折计算 $\log \pi_\phi(a_i\mid s_i)$。

**第四，一定要 clip 权重。**  
推荐：

$$
w_{\min}=0.2,\quad w_{\max}=3\sim 10
$$
如果动作维度高，先用：

$$
w_{\max}=3 \text{ 或 } 5
$$
不要一开始用太大。

**第五，监控有效样本数。**

$$
N_{\text{eff}}
=
\frac{
(\sum_i w_i)^2
}{
\sum_i w_i^2
}
$$
如果：

$$
N_{\text{eff}}/N
$$
太小，说明训练被少量低似然动作支配了。通常希望它至少不要低于：

$$
0.1\sim 0.2
$$
---

### 4.9 最终推荐做法

不建议直接用：

$$
w_i=\exp(-\beta\log\pi(a_i\mid s_i))
$$
而是用下面这个版本。

如果行为策略是普通 diagonal Gaussian：

$$
\boxed{
w_i
=
(1-\beta)^{d_a/2}
\exp
\left(
\frac{\beta}{2}
\sum_k
\left(
\frac{a_{i,k}-\mu_k(s_i)}
{\sigma_k(s_i)}
\right)^2
\right)
}
$$
然后：

$$
w_i\leftarrow \text{clip}(w_i,w_{\min},w_{\max})
$$
$$
w_i\leftarrow \frac{w_i}{\text{mean}(w)}
$$
推荐初始参数：

$$
\beta=0.2\sim 0.5
$$
如果动作维度较高，先用：

$$
\beta=0.1\sim 0.3
$$
如果行为策略是 tanh Gaussian、MDN、flow 或其他密度模型，则用：

$$
\boxed{
w_i
=
\frac{
\exp(-\beta\log \pi_\phi(a_i\mid s_i))
}{
Z(s_i)
}
}
$$
其中 $Z(s_i)$ 用状态邻域局部均值近似。

最终 FM loss：

$$
\boxed{
\mathcal L
=
\frac{
\sum_i
w_i
\left\|
v_\theta(t,x_t,s_i)-u_t
\right\|^2
}{
\sum_i w_i
}
}
$$
结论是：**方法合理，但要把 $\log\pi(a\mid s)$ 当作条件密度估计来用，并且必须做条件归一化、温度控制和权重裁剪。** 对单高斯策略，最推荐用 Mahalanobis 距离形式，而不是直接用原始 $-\log\pi(a\mid s)$。

---

## 5. 本轮讨论的核心结论汇总

### 5.1 加权 Flow Matching 的本质

加权 FM 不是“更准确地拟合原始数据分布”，而是拟合：

$$
q(a\mid s)\propto w(s,a)p_{\text{data}}(a\mid s)
$$
因此它是一种主动重塑目标条件分布的方法。

---

### 5.2 平衡 $p(a\mid s)$ 的关键约束

为了只改变动作分布、不改变状态分布，权重应满足：

$$
\mathbb E[w(s,a)\mid s]\approx 1
$$
这就是条件归一化的意义。

---

### 5.3 不推荐直接全局反密度加权

全局 $w(a)$ 容易把状态分布也改掉，并且可能放大噪声动作。

---

### 5.4 不想聚类时的替代

最稳的无聚类方案是：

$$
\boxed{
\text{状态 kNN 邻域} + \text{动作 kNN 稀疏度} + \text{局部归一化}
}
$$
---

### 5.5 使用高斯行为策略是合理的

用高斯策略估计 $\pi(a\mid s)$，再根据 likelihood 设权重是合理的，但推荐使用 Mahalanobis 距离形式：

$$
w_i
=
(1-\beta)^{d_a/2}
\exp
\left(
\frac{\beta}{2}r_i^2
\right)
$$
而不是直接：

$$
w_i=\exp(-\beta\log\pi(a_i\mid s_i))
$$
因为后者会把状态相关的方差项也引入权重，可能改变 $p(s)$。

---

### 5.6 训练实现原则

始终用：

$$
\mathcal L
=
\frac{
\sum_i w_i
\left\|
v_\theta(t,x_t,s_i)-u_t
\right\|^2
}{
\sum_i w_i
}
$$
不要用：

$$
v_\theta \approx w_i u_t
$$
权重应该乘在 loss 上，而不是乘在速度目标上。

---



