# 使用 tanh Gaussian 行为策略计算 Flow Matching 加权损失权重

## 问题

现在需要通过 **tanh Gaussian** 建模行为策略，然后对数据集的所有动作计算未归一化的 log-weight，再在全数据集上归一化，随后进行 clipping，主要用于避免出现极小的权重。最终权重作为 Flow Matching 的损失权重。

---

## 1. 数据集定义

设离线数据集为：

$$
\mathcal D=\{(s_i,a_i)\}_{i=1}^N
$$

其中：

$$
s_i \in \mathcal S,\qquad a_i\in\mathcal A
$$

动作维度为：

$$
d_a
$$

---

## 2. tanh Gaussian 行为策略

定义 pre-tanh 隐变量：

$$
u\sim \mathcal N\left(
\mu_\phi(s),
\operatorname{diag}(\sigma_\phi^2(s))
\right)
$$

经过 tanh 映射得到归一化动作：

$$
y=\tanh(u)
$$

如果环境动作范围为：

$$
a\in [a_{\min},a_{\max}]^{d_a}
$$

则真实动作由如下变换得到：

$$
a=c\odot y+b
$$

其中：

$$
c=\frac{a_{\max}-a_{\min}}{2}
$$

$$
b=\frac{a_{\max}+a_{\min}}{2}
$$

如果动作已经归一化到：

$$
[-1,1]^{d_a}
$$

则有：

$$
c=\mathbf 1
$$

$$
b=\mathbf 0
$$

---

## 3. 将数据动作映射到 pre-tanh 空间

对于数据集中的动作 $a_i$，先映射到 tanh 输出空间：

$$
y_i=\frac{a_i-b}{c}
$$

为了避免 $\operatorname{atanh}(\pm 1)$ 数值发散，进行裁剪：

$$
y_i\leftarrow \operatorname{clip}(y_i,-1+\epsilon,1-\epsilon)
$$

然后反变换到 pre-tanh 空间：

$$
u_i=\operatorname{atanh}(y_i)
$$

其中：

$$
\operatorname{atanh}(y_i)
=
\frac{1}{2}
\log
\frac{1+y_i}{1-y_i}
$$

---

## 4. tanh Gaussian 的 log-density

行为策略在数据动作上的 log-density 为：

$$
\log \pi_\phi(a_i\mid s_i)
=
\log \mathcal N
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
\log c_k
$$

其中 Gaussian 部分为：

$$
\log \mathcal N
\left(
u_i;
\mu_i,
\operatorname{diag}(\sigma_i^2)
\right)
=
-\frac{1}{2}
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
\mu_i=\mu_\phi(s_i)
$$

$$
\sigma_i=\sigma_\phi(s_i)
$$

如果动作已经处于 $[-1,1]^{d_a}$，则 $c_k=1$，对应的尺度修正项为：

$$
\sum_{k=1}^{d_a}\log c_k=0
$$

---

## 5. 定义未归一化 log-weight

定义未归一化 log-weight：

$$
\widetilde{\ell}_i
=
-\beta
\log \pi_\phi(a_i\mid s_i)
$$

其中：

$$
\beta\in[0,1)
$$

控制反密度加权强度。

等价地，未归一化 weight 为：

$$
\widetilde w_i
=
\exp(\widetilde{\ell}_i)
$$

即：

$$
\widetilde w_i
=
\pi_\phi(a_i\mid s_i)^{-\beta}
$$

当 $\log \pi_\phi(a_i\mid s_i)$ 越小，说明该动作在行为策略下越低密度，此时 $\widetilde w_i$ 越大。

---

## 6. 全数据集归一化

在全数据集上计算归一化常数：

$$
\log \bar Z
=
\log
\left(
\frac{1}{N}
\sum_{j=1}^{N}
\exp(\widetilde{\ell}_j)
\right)
$$

归一化后的 log-weight 为：

$$
\ell_i^{\mathrm{norm}}
=
\widetilde{\ell}_i-\log \bar Z
$$

对应的归一化 weight 为：

$$
w_i^{\mathrm{norm}}
=
\exp
\left(
\ell_i^{\mathrm{norm}}
\right)
$$

也就是：

$$
w_i^{\mathrm{norm}}
=
\frac{
\exp(\widetilde{\ell}_i)
}{
\frac{1}{N}
\sum_{j=1}^{N}
\exp(\widetilde{\ell}_j)
}
$$

因此，归一化后的权重满足：

$$
\frac{1}{N}
\sum_{i=1}^{N}
w_i^{\mathrm{norm}}
=
1
$$

---

## 7. Weight clipping

为了避免权重过小或过大，对归一化后的权重进行 clipping。

log-space 形式为：

$$
\ell_i^{\mathrm{clip}}
=
\operatorname{clip}
\left(
\ell_i^{\mathrm{norm}},
\log w_{\min},
\log w_{\max}
\right)
$$

最终权重为：

$$
w_i
=
\exp
\left(
\ell_i^{\mathrm{clip}}
\right)
$$

等价地，可以直接在 weight-space 中写为：

$$
w_i
=
\operatorname{clip}
\left(
w_i^{\mathrm{norm}},
w_{\min},
w_{\max}
\right)
$$

如果主要目的是避免极小权重，也可以只设置下界：

$$
w_i
=
\max
\left(
w_i^{\mathrm{norm}},
w_{\min}
\right)
$$

如果希望 clipping 后仍然保持平均权重为 1，可以再次归一化：

$$
w_i
\leftarrow
\frac{
w_i
}{
\frac{1}{N}
\sum_{j=1}^{N}
w_j
}
$$

---

## 8. Weighted Flow Matching 损失

标准 Flow Matching 中，采样噪声端点：

$$
a_0\sim p_0(a_0)
$$

采样时间：

$$
t\sim \mathcal U(0,1)
$$

线性插值得到：

$$
x_t=(1-t)a_0+t a_i
$$

对应目标速度为：

$$
u_t(a_0,a_i)=a_i-a_0
$$

使用样本权重后的 Flow Matching 损失为：

$$
\mathcal L_{\mathrm{wFM}}
=
\frac{
\sum_{i=1}^{N}
w_i
\mathbb E_{t,a_0}
\left[
\left\|
v_\theta(t,x_t,s_i)
-
u_t(a_0,a_i)
\right\|_2^2
\right]
}{
\sum_{i=1}^{N}
w_i
}
$$

将线性路径目标速度代入，有：

$$
\mathcal L_{\mathrm{wFM}}
=
\frac{
\sum_{i=1}^{N}
w_i
\mathbb E_{t,a_0}
\left[
\left\|
v_\theta(t,x_t,s_i)
-
(a_i-a_0)
\right\|_2^2
\right]
}{
\sum_{i=1}^{N}
w_i
}
$$

---

## 9. 整体流程总结

整体流程可以写成：

$$
(s_i,a_i)
\longrightarrow
\log \pi_\phi(a_i\mid s_i)
$$

$$
\log \pi_\phi(a_i\mid s_i)
\longrightarrow
\widetilde{\ell}_i
=
-\beta
\log \pi_\phi(a_i\mid s_i)
$$

$$
\widetilde{\ell}_i
\longrightarrow
\ell_i^{\mathrm{norm}}
=
\widetilde{\ell}_i
-
\log
\left(
\frac{1}{N}
\sum_{j=1}^{N}
\exp(\widetilde{\ell}_j)
\right)
$$

$$
\ell_i^{\mathrm{norm}}
\longrightarrow
w_i
=
\operatorname{clip}
\left(
\exp(\ell_i^{\mathrm{norm}}),
w_{\min},
w_{\max}
\right)
$$

$$
w_i
\longrightarrow
\mathcal L_{\mathrm{wFM}}
=
\frac{
\sum_{i=1}^{N}
w_i
\mathbb E_{t,a_0}
\left[
\left\|
v_\theta(t,x_t,s_i)
-
u_t(a_0,a_i)
\right\|_2^2
\right]
}{
\sum_{i=1}^{N}
w_i
}
$$

---

## 10. 简洁版公式

如果只保留核心公式，可以写为：

$$
\widetilde{\ell}_i
=
-\beta
\log \pi_\phi(a_i\mid s_i)
$$

$$
\ell_i^{\mathrm{norm}}
=
\widetilde{\ell}_i
-
\log
\left(
\frac{1}{N}
\sum_{j=1}^{N}
e^{\widetilde{\ell}_j}
\right)
$$

$$
w_i
=
\operatorname{clip}
\left(
e^{\ell_i^{\mathrm{norm}}},
w_{\min},
w_{\max}
\right)
$$

$$
\mathcal L_{\mathrm{wFM}}
=
\frac{
\sum_{i=1}^{N}
w_i
\mathbb E_{t,a_0}
\left[
\left\|
v_\theta(t,x_t,s_i)
-
u_t(a_0,a_i)
\right\|_2^2
\right]
}{
\sum_{i=1}^{N}
w_i
}
$$
