# DDPM 扩散概率模型详解

> 主题：Denoising Diffusion Probabilistic Models（DDPM）  
> 核心问题：前向扩散为何有闭式解、真实一步反向后验如何推导、ELBO 如何拆成 KL、KL 又如何变成预测噪声的 MSE。  
> 阅读目标：不仅记住公式，还能说清每一次代入、配方、约去常数和符号切换的理由。

## 资料与范围

- 学习材料：[与 ChatGPT 的 DDPM 学习对话](https://chatgpt.com/share/6a8fefe7-9a60-83e8-8013-f69e0d67e002)
- 原始论文：Jonathan Ho, Ajay Jain, Pieter Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020
- 本文只解释 DDPM 的概率模型、前向过程、反向过程、ELBO 与训练/采样算法。

## 先看完整主线

DDPM 的逻辑可以压缩为四层，但每一层的对象不同：

1. 人为规定一个前向马尔可夫链 $q$，逐步把数据变成高斯噪声。
2. 利用线性高斯结构，解析求出训练时可用的真实一步反向后验。
3. 用神经网络定义反向模型 $p_\theta$，让它逼近真实后验。
4. 从最大似然得到 ELBO，再由高斯 KL 得到加权噪声 MSE，最后采用简化噪声 MSE 训练。

![DDPM 前向扩散与学习到的反向去噪](assets/ddpm/overview.png)

*图：上方表示人为规定的前向扩散 $q$，下方表示模型学习的反向过程 $p_\theta$。图片只表达总体方向，不代表某一次真实实验的中间结果。AI 生成示意图。*

整条推导链为：

$$\mathrm{Maximum\ Likelihood}\rightarrow \mathrm{ELBO}\rightarrow D_{\mathrm{KL}}\bigl(q\|p_\theta\bigr)\rightarrow \text{weighted noise MSE}\rightarrow L_{\mathrm{simple}}$$

最后一个箭头是训练目标的重新加权简化，不是恒等变形。这一点后文会严格说明。

## 最重要的阅读约定：取值与分布不是一回事

推导中经常在下面两种表达之间切换：

- 条件分布：$q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I)$；
- 从该分布得到样本的表达式：$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t$。

第一行回答“随机变量服从什么分布”，第二行回答“怎样用标准高斯噪声构造一个该分布的样本”。二者依靠高斯重参数化相连，但不能把条件分布 $q(\cdot)$ 和随机变量 $x_t$ 当成同一个对象。

下文每次出现“分布式”和“采样式”都会主动说明。

## 符号表与责任边界

| 符号 | 含义 | 是否学习 | 属于哪个过程 |
|---|---|---:|---|
| $x_0\in\mathbb R^d$ | 一条真实数据，例如展平后的图像 | 否 | 数据分布 |
| $x_t\in\mathbb R^d$ | 第 $t$ 步的带噪变量 | 否 | 前向或反向链中的状态 |
| $T$ | 总扩散步数 | 否 | 超参数 |
| $\beta_t\in(0,1)$ | 第 $t$ 步前向噪声方差 | 通常否 | 前向过程 $q$ |
| $\alpha_t=1-\beta_t$ | 第 $t$ 步信号保留系数 | 否 | 前向过程 $q$ |
| $\bar\alpha_t=\prod_{s=1}^t\alpha_s$ | 前 $t$ 步累计信号系数 | 否 | 前向过程 $q$ |
| $\epsilon_t\sim\mathcal N(0,I)$ | 第 $t$ 个前向转移中新采样的噪声 | 否 | 单步采样式 |
| $\epsilon\sim\mathcal N(0,I)$ | 把前 $t$ 步所有独立噪声合并后的等价标准噪声 | 否 | 闭式采样式 |
| $q$ | 人为规定的前向过程及由它推出的后验 | 否 | 已知过程 |
| $\tilde\mu_t$ | 真实一步反向后验的均值 | 否 | 解析后验 $q$ |
| $\tilde\beta_t$ | 真实一步反向后验的方差系数 | 否 | 解析后验 $q$ |
| $p_\theta$ | 神经网络参数化的反向生成过程 | 是 | 学习过程 |
| $\mu_\theta(x_t,t)$ | 模型反向分布的均值 | 是 | 模型 $p_\theta$ |
| $\sigma_t^2$ | 模型反向分布采用的方差 | 固定或学习 | 模型 $p_\theta$ |
| $\epsilon_\theta(x_t,t)$ | 网络对闭式采样噪声 $\epsilon$ 的预测 | 是 | 噪声参数化 |

必须特别区分：$\tilde\beta_t$ 是由前向过程解析推出的真实后验方差；$\sigma_t^2$ 是反向模型采用的方差设定。它们可以被设成相同，但概念上不是同一个量。

## 预备数学一：连乘与累计系数

连乘符号的定义是：

$$\prod_{s=1}^{t}\alpha_s=\alpha_1\alpha_2\cdots\alpha_t$$

DDPM 定义：

$$\bar\alpha_t:=\prod_{s=1}^{t}\alpha_s$$

这里的横线不是求平均，而只是“累计乘积”的约定记号。因此：

$$\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$$

由于每个 $\alpha_t\in(0,1)$，累计乘积通常随 $t$ 增大而减小。

## 预备数学二：高斯变量的仿射变换

若 $z\sim\mathcal N(0,I)$，则：

$$x=\mu+Lz\sim\mathcal N(\mu,LL^\top)$$

特别地，若 $L=\sigma I$，则：

$$x=\mu+\sigma z\sim\mathcal N(\mu,\sigma^2 I)$$

这就是“分布式”与“采样式”之间的桥梁。

## 预备数学三：独立高斯噪声为何能够合并

令 $u,v\sim\mathcal N(0,I)$ 且相互独立。对标量 $a,b$：

$$au+bv\sim\mathcal N\bigl(0,(a^2+b^2)I\bigr)$$

均值为零，因为：

$$\mathbb E[au+bv]=a\mathbb E[u]+b\mathbb E[v]=0$$

协方差逐步展开为：

$$\mathrm{Cov}(au+bv)=a^2\mathrm{Cov}(u)+b^2\mathrm{Cov}(v)+ab\mathrm{Cov}(u,v)+ab\mathrm{Cov}(v,u)$$

独立意味着交叉协方差为零，而 $\mathrm{Cov}(u)=\mathrm{Cov}(v)=I$，所以：

$$\mathrm{Cov}(au+bv)=(a^2+b^2)I$$

反过来，若某个零均值高斯变量的协方差是 $cI$，便可在分布意义上写成 $\sqrt c\,\epsilon$，其中 $\epsilon\sim\mathcal N(0,I)$。

注意“在分布意义上”等价不表示原来的多个噪声样本在数值上突然变成了同一个旧样本；它表示可以定义一个新的标准高斯变量 $\epsilon$，使两边同分布。

## 第一部分：前向扩散过程

### 单步条件分布

DDPM 人为规定前向马尔可夫链：

$$q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1})$$

每一步的条件分布为：

$$q(x_t\mid x_{t-1})=\mathcal N\bigl(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I\bigr)$$

分号前的 $x_t$ 表示这个高斯密度正在评价的变量；分号后的两项分别是均值与协方差。等价的采样式为：

$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t,\qquad \epsilon_t\sim\mathcal N(0,I)$$

因为 $\alpha_t=1-\beta_t$，也可写成：

$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t$$

### 从第一步展开到第二步

第一步：

$$x_1=\sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}\epsilon_1$$

第二步原式：

$$x_2=\sqrt{\alpha_2}x_1+\sqrt{1-\alpha_2}\epsilon_2$$

把 $x_1$ 代入：

$$x_2=\sqrt{\alpha_2}\left(\sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}\epsilon_1\right)+\sqrt{1-\alpha_2}\epsilon_2$$

分配 $\sqrt{\alpha_2}$：

$$x_2=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{\alpha_2(1-\alpha_1)}\epsilon_1+\sqrt{1-\alpha_2}\epsilon_2$$

信号项已经是：

$$\sqrt{\alpha_1\alpha_2}x_0=\sqrt{\bar\alpha_2}x_0$$

后两项是两个独立零均值高斯变量之和。它们的总协方差系数为：

$$\alpha_2(1-\alpha_1)+(1-\alpha_2)=\alpha_2-\alpha_1\alpha_2+1-\alpha_2=1-\alpha_1\alpha_2$$

因为 $\bar\alpha_2=\alpha_1\alpha_2$，所以可定义新的 $\epsilon\sim\mathcal N(0,I)$，在分布意义上写为：

$$x_2=\sqrt{\bar\alpha_2}x_0+\sqrt{1-\bar\alpha_2}\epsilon$$

### 用归纳法推广到任意时刻

假设在 $t-1$ 时结论成立：

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}}\epsilon',\qquad \epsilon'\sim\mathcal N(0,I)$$

第 $t$ 步为：

$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t$$

代入归纳假设：

$$x_t=\sqrt{\alpha_t\bar\alpha_{t-1}}x_0+\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\epsilon'+\sqrt{1-\alpha_t}\epsilon_t$$

由 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$，信号项变为：

$$\sqrt{\alpha_t\bar\alpha_{t-1}}x_0=\sqrt{\bar\alpha_t}x_0$$

两个独立噪声项的总方差系数为：

$$\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\bar\alpha_{t-1}=1-\bar\alpha_t$$

因此可再次用一个新的标准高斯变量 $\epsilon$ 表示合并后的噪声：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$

这是一条“样本构造式”。对应的“条件分布式”为：

$$q(x_t\mid x_0)=\mathcal N\bigl(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\bigr)$$

### 闭式解为什么重要

训练时不需要真的依次生成 $x_1,x_2,\ldots,x_t$。只需：

1. 从数据集中取 $x_0$；
2. 随机抽取 $t$；
3. 抽取一次 $\epsilon\sim\mathcal N(0,I)$；
4. 用闭式公式直接构造 $x_t$。

这使不同训练样本和不同时间步可以并行生成。

### 终点为什么只是近似标准高斯

由闭式分布：

$$q(x_T\mid x_0)=\mathcal N\bigl(\sqrt{\bar\alpha_T}x_0,(1-\bar\alpha_T)I\bigr)$$

若噪声日程使 $\bar\alpha_T\approx 0$，则：

$$q(x_T\mid x_0)\approx\mathcal N(0,I)$$

这里通常是近似关系，不应在有限 $T$ 下无条件写成严格相等。

## 第二部分：真实一步反向后验

### 为什么求的是带有原始数据条件的后验

生成时真正希望知道 $q(x_{t-1}\mid x_t)$，但该分布需要对未知数据分布积分，一般没有可直接使用的解析式。

训练时 $x_0$ 已知，因此可以精确计算：

$$q(x_{t-1}\mid x_t,x_0)$$

它是“前向过程诱导出的真实一步反向后验”，不是神经网络模型。

### 从条件 Bayes 公式开始

条件 Bayes 公式给出严格等式：

$$q(x_{t-1}\mid x_t,x_0)=\frac{q(x_t\mid x_{t-1},x_0)q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}$$

前向过程具有马尔可夫性质：给定 $x_{t-1}$ 后，$x_t$ 与更早的 $x_0$ 条件独立，因此：

$$q(x_t\mid x_{t-1},x_0)=q(x_t\mid x_{t-1})$$

代回仍然是严格等式：

$$q(x_{t-1}\mid x_t,x_0)=\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}$$

现在把 $x_t$ 和 $x_0$ 当作已经给定的条件，只把 $x_{t-1}$ 当作待研究变量。分母 $q(x_t\mid x_0)$ 不随 $x_{t-1}$ 改变，所以在求关于 $x_{t-1}$ 的分布形状时，可吸收到归一化常数中：

$$q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)$$

这里从 $=$ 变成 $\propto$，丢掉的不是“数值上为零的项”，而是所有与当前待求变量 $x_{t-1}$ 无关的正比例因子。最后仍需归一化，才能重新成为概率密度。

### 写出两个高斯因子

为缩短记号，暂令 $y=x_{t-1}$。第一个因子为：

$$q(x_t\mid y)=\mathcal N\bigl(x_t;\sqrt{\alpha_t}y,\beta_t I\bigr)$$

第二个因子由前向闭式解得到：

$$q(y\mid x_0)=\mathcal N\bigl(y;\sqrt{\bar\alpha_{t-1}}x_0,(1-\bar\alpha_{t-1})I\bigr)$$

忽略高斯密度前面与 $y$ 无关的归一化系数，只保留指数中依赖 $y$ 的部分：

$$q(y\mid x_t,x_0)\propto\exp\left[-\frac{\|x_t-\sqrt{\alpha_t}y\|^2}{2\beta_t}-\frac{\|y-\sqrt{\bar\alpha_{t-1}}x_0\|^2}{2(1-\bar\alpha_{t-1})}\right]$$

### 逐项展开平方

第一项的平方为：

$$\|x_t-\sqrt{\alpha_t}y\|^2=x_t^\top x_t-2\sqrt{\alpha_t}x_t^\top y+\alpha_t y^\top y$$

第二项的平方为：

$$\|y-\sqrt{\bar\alpha_{t-1}}x_0\|^2=y^\top y-2\sqrt{\bar\alpha_{t-1}}x_0^\top y+\bar\alpha_{t-1}x_0^\top x_0$$

$x_t^\top x_t$ 与 $x_0^\top x_0$ 不含 $y$，所以它们只改变归一化常数，不改变后验关于 $y$ 的均值和方差。收集 $y^\top y$ 与线性项：

$$\log q(y\mid x_t,x_0)=C-\frac{1}{2}\left[A_ty^\top y-2B_t^\top y\right]$$

其中 $C$ 与 $y$ 无关，而：

$$A_t=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}$$

$$B_t=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0$$

### 完成平方

向量形式的配方恒等式为：

$$A_ty^\top y-2B_t^\top y=A_t\left\|y-\frac{B_t}{A_t}\right\|^2-\frac{B_t^\top B_t}{A_t}$$

最后一项不含 $y$，继续并入常数。于是后验一定是高斯：

$$q(y\mid x_t,x_0)=\mathcal N\left(y;\frac{B_t}{A_t},A_t^{-1}I\right)$$

这就严格解释了“两个关于同一变量的高斯密度相乘，归一化后仍是高斯”：二者的负对数都是二次函数，相加后仍是二次函数，配方后仍具有高斯核的形式。

### 计算真实后验方差

先对 $A_t$ 通分：

$$A_t=\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}$$

利用 $\beta_t=1-\alpha_t$：

$$\alpha_t(1-\bar\alpha_{t-1})+\beta_t=\alpha_t-\alpha_t\bar\alpha_{t-1}+1-\alpha_t=1-\bar\alpha_t$$

因此：

$$A_t=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}$$

取倒数得到后验方差系数：

$$\tilde\beta_t:=A_t^{-1}=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

### 计算真实后验均值

后验均值为 $B_t/A_t=\tilde\beta_tB_t$：

$$\tilde\mu_t(x_t,x_0)=\tilde\beta_t\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\right)$$

分别化简 $x_t$ 和 $x_0$ 的系数：

$$\tilde\beta_t\frac{\sqrt{\alpha_t}}{\beta_t}=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

$$\tilde\beta_t\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}$$

所以：

$$\tilde\mu_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

最终真实一步反向后验为：

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N\bigl(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I\bigr)$$

## 第三部分：从真实后验到学习到的反向模型

### 为什么不能直接拿真实后验生成

$\tilde\mu_t(x_t,x_0)$ 依赖真实数据 $x_0$。训练时我们从数据集中拿到了 $x_0$，但生成时从 $x_T\sim\mathcal N(0,I)$ 出发，根本没有真实 $x_0$ 可用。

因此 DDPM 定义一个神经网络参数化的反向马尔可夫链：

$$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)$$

其中：

$$p(x_T)=\mathcal N(0,I)$$

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N\bigl(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I\bigr)$$

此处发生了对象切换：

- $q(x_{t-1}\mid x_t,x_0)$ 是已知前向过程诱导出的真实后验；
- $p_\theta(x_{t-1}\mid x_t)$ 是生成时实际使用、需要学习的模型分布；
- $\tilde\beta_t$ 是前者的解析方差；
- $\sigma_t^2$ 是后者选择或学习的方差。

### 用闭式采样式消去原始数据

前向闭式采样式为：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

先移项：

$$\sqrt{\bar\alpha_t}x_0=x_t-\sqrt{1-\bar\alpha_t}\epsilon$$

再除以 $\sqrt{\bar\alpha_t}$：

$$x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}$$

把它代入 $\tilde\mu_t(x_t,x_0)$。为避免跳步，先只代入而不整理：

$$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

由 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$：

$$\frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}}=\frac{1}{\sqrt{\alpha_t}}$$

所以第一大项变成：

$$\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)}x_t-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\epsilon$$

现在合并两个 $x_t$ 系数。提取公共分母 $\sqrt{\alpha_t}(1-\bar\alpha_t)$：

$$\frac{\beta_t+\alpha_t(1-\bar\alpha_{t-1})}{\sqrt{\alpha_t}(1-\bar\alpha_t)}$$

分子化简为：

$$\beta_t+\alpha_t(1-\bar\alpha_{t-1})=1-\alpha_t+\alpha_t-\alpha_t\bar\alpha_{t-1}=1-\bar\alpha_t$$

所以 $x_t$ 的总系数就是 $1/\sqrt{\alpha_t}$。最终：

$$\tilde\mu_t(x_t,\epsilon)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$$

这不是把 $x_0$ “随意改名”为 $x_t$，而是先利用前向闭式关系把 $x_0$ 解出来，再做一次严格代入和代数化简。

生成时真实 $\epsilon$ 未知，因此网络预测：

$$\epsilon_\theta(x_t,t)\approx\epsilon$$

并据此参数化模型均值：

$$\mu_\theta(x_t,t):=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

因此“预测噪声”并不是一句经验性的“预测后把噪声减掉”，而是因为真实后验均值能够被精确改写为 $x_t$ 与噪声 $\epsilon$ 的函数。

## 第四部分：ELBO 从哪里来

### 最大似然目标

生成模型希望给真实数据较高概率，因此目标是最大化：

$$\log p_\theta(x_0)$$

但 DDPM 的中间变量 $x_{1:T}$ 没有被观测，需要积分掉：

$$p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}$$

这是高维积分，通常无法直接求值。

### 一般隐变量模型中的 Jensen 推导

先考虑一般隐变量 $z$。在 $q(z\mid x)>0$ 覆盖目标联合分布所需区域的条件下：

$$\log p_\theta(x)=\log\int p_\theta(x,z)\,dz$$

在积分内乘除同一个辅助分布 $q(z\mid x)$：

$$\log p_\theta(x)=\log\int q(z\mid x)\frac{p_\theta(x,z)}{q(z\mid x)}\,dz$$

积分正是关于 $q$ 的期望：

$$\log p_\theta(x)=\log\mathbb E_{q(z\mid x)}\left[\frac{p_\theta(x,z)}{q(z\mid x)}\right]$$

$\log$ 是凹函数，Jensen 不等式给出：

$$\log\mathbb E_q[Y]\geq\mathbb E_q[\log Y]$$

令 $Y=p_\theta(x,z)/q(z\mid x)$，得到：

$$\log p_\theta(x)\geq\mathbb E_{q(z\mid x)}\left[\log p_\theta(x,z)-\log q(z\mid x)\right]$$

右侧定义为证据下界：

$$\mathcal L_{\mathrm{ELBO}}:=\mathbb E_{q(z\mid x)}\left[\log p_\theta(x,z)-\log q(z\mid x)\right]$$

### 下界与真实似然之间为何是 KL

由 Bayes 关系：

$$p_\theta(x,z)=p_\theta(z\mid x)p_\theta(x)$$

代入 ELBO：

$$\mathcal L_{\mathrm{ELBO}}=\mathbb E_q\left[\log p_\theta(z\mid x)+\log p_\theta(x)-\log q(z\mid x)\right]$$

$\log p_\theta(x)$ 不依赖 $z$，所以其期望仍是自身：

$$\mathcal L_{\mathrm{ELBO}}=\log p_\theta(x)-\mathbb E_q\left[\log\frac{q(z\mid x)}{p_\theta(z\mid x)}\right]$$

括号中的期望正是 KL 散度：

$$\log p_\theta(x)=\mathcal L_{\mathrm{ELBO}}+D_{\mathrm{KL}}\bigl(q(z\mid x)\|p_\theta(z\mid x)\bigr)$$

由于 KL 非负，ELBO 不超过对数似然。只有当辅助后验与模型真实后验几乎处处相等时，KL 为零、下界变紧。

## 第五部分：DDPM 的 ELBO 如何拆开

### 把整条前向轨迹当作隐变量

在 DDPM 中令 $z=x_{1:T}$，辅助分布就是已知的前向链 $q(x_{1:T}\mid x_0)$。负 ELBO 写为：

$$L_{\mathrm{VLB}}=\mathbb E_q\left[\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}\right]$$

它是负对数似然的上界：

$$-\log p_\theta(x_0)\leq L_{\mathrm{VLB}}$$

反向模型联合分布已知：

$$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)$$

接下来需要把前向链也改写成适合逐步比较的反向后验乘积。

### 前向链的反向后验分解

对任意 $t\geq2$，条件 Bayes 给出：

$$q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)=q(x_{t-1}\mid x_t,x_0)q(x_t\mid x_0)$$

整理：

$$q(x_t\mid x_{t-1})=q(x_{t-1}\mid x_t,x_0)\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}$$

把 $t=2$ 到 $T$ 的这些比值连乘时，中间的边缘分布会望远镜式消去，最终得到：

$$q(x_{1:T}\mid x_0)=q(x_T\mid x_0)\prod_{t=2}^{T}q(x_{t-1}\mid x_t,x_0)$$

将 $q$ 与 $p_\theta$ 的两个分解代入负 ELBO：

$$L_{\mathrm{VLB}}=\mathbb E_q\left[\log\frac{q(x_T\mid x_0)}{p(x_T)}+\sum_{t=2}^{T}\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}-\log p_\theta(x_0\mid x_1)\right]$$

于是可以写成三类项：

$$L_{\mathrm{VLB}}=L_T+\sum_{t=2}^{T}L_{t-1}+L_0$$

终点先验匹配项：

$$L_T=D_{\mathrm{KL}}\bigl(q(x_T\mid x_0)\|p(x_T)\bigr)$$

中间每一步的反向匹配项：

$$L_{t-1}=D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)\bigr)$$

最后的数据重建项：

$$L_0=-\log p_\theta(x_0\mid x_1)$$

这说明 DDPM 中的 KL 不是凭空添加的正则项。它来自整条隐变量生成链的负 ELBO 分解，要求模型的一步反向分布逼近由前向过程解析得到的真实一步后验。

## 第六部分：从高斯 KL 到噪声 MSE

### 两个反向分布各自是什么

真实后验：

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I)$$

模型分布：

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta,\sigma_t^2I)$$

当模型方差 $\sigma_t^2$ 被固定、不依赖 $\theta$ 时，高斯 KL 中所有与均值网络参数有关的部分为：

$$L_{t-1}=\mathbb E_q\left[\frac{1}{2\sigma_t^2}\|\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\|^2\right]+C$$

$C$ 可以依赖时间步和既定方差，但不依赖均值网络参数 $\theta$，因此优化均值网络时可忽略。

### 两个均值都写成噪声参数化

真实后验均值：

$$\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$$

模型均值：

$$\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

两式相减时 $x_t$ 精确抵消：

$$\tilde\mu_t-\mu_\theta=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\left(\epsilon-\epsilon_\theta(x_t,t)\right)$$

取平方范数：

$$\|\tilde\mu_t-\mu_\theta\|^2=\frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)}\|\epsilon-\epsilon_\theta(x_t,t)\|^2$$

代回 KL 的均值项：

$$L_{t-1}=\mathbb E_q\left[w_t\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]+C$$

其中：

$$w_t=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$$

到这里为止，噪声 MSE 是带有时间权重 $w_t$ 的。

### 简化损失与完整变分目标的关系

经典 DDPM 常使用：

$$L_{\mathrm{simple}}=\mathbb E_{x_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]$$

其中：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

$L_{\mathrm{simple}}$ 去掉了完整变分目标中随时间变化的权重。因此更准确的表述是：

- 完整负 ELBO 严格导出的是带权噪声 MSE，以及终点项和重建项；
- $L_{\mathrm{simple}}$ 是作者采用的重新加权简化目标；
- 它与完整 ELBO 有紧密理论联系，但二者不能在省略权重后仍写成严格相等。

## 第七部分：训练算法逐步对应公式

一次训练迭代可以写成：

1. 采样真实数据 $x_0\sim q_{\mathrm{data}}$。
2. 均匀采样时间步 $t\sim\mathrm{Uniform}\{1,\ldots,T\}$。
3. 采样 $\epsilon\sim\mathcal N(0,I)$。
4. 用闭式前向公式构造 $x_t$。
5. 网络接收 $(x_t,t)$ 并输出 $\epsilon_\theta(x_t,t)$。
6. 计算真实噪声与预测噪声的平方误差并更新 $\theta$。

对应的伪代码为：

```text
repeat
    x0  <- sample_data()
    t   <- random_integer(1, T)
    eps <- standard_normal(shape(x0))

    xt <- sqrt(alpha_bar[t]) * x0
          + sqrt(1 - alpha_bar[t]) * eps

    eps_pred <- network(xt, t)
    loss <- mean_squared_error(eps, eps_pred)
    update_theta(loss)
until convergence
```

网络学习的是“给定当前带噪状态和噪声等级，估计产生该状态的等价噪声”。时间步 $t$ 必须作为条件输入，因为相同的像素值在不同噪声强度下应采用不同的去噪策略。

## 第八部分：采样算法逐步对应公式

生成从标准高斯开始：

$$x_T\sim\mathcal N(0,I)$$

对 $t=T,T-1,\ldots,1$，先计算模型均值：

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

再从模型反向条件分布采样：

$$x_{t-1}=\mu_\theta(x_t,t)+\sigma_tz,qquad z\sim\mathcal N(0,I)$$

在最后一步通常不再额外注入随机噪声，即令 $t=1$ 时的 $z=0$。伪代码为：

```text
x <- standard_normal(data_shape)

for t = T, T-1, ..., 1
    eps_pred <- network(x, t)
    mean <- (x - beta[t] / sqrt(1 - alpha_bar[t]) * eps_pred)
            / sqrt(alpha[t])

    if t > 1
        z <- standard_normal(shape(x))
    else
        z <- 0

    x <- mean + sigma[t] * z

return x
```

这里的随机项并非“模型又把刚去掉的噪声加回去”。它来自模型对 $p_\theta(x_{t-1}\mid x_t)$ 的概率分布建模，用来表达同一个 $x_t$ 可能对应多个合理的较干净状态。

## 常见混淆逐项澄清

### 当前步噪声与累计等价噪声

$\epsilon_t$ 是从 $x_{t-1}$ 到 $x_t$ 的单步新噪声；闭式公式中的 $\epsilon$ 是前 $t$ 步所有独立噪声线性组合后重新标准化得到的等价噪声。二者都服从标准高斯，但语义不同。

### 真实后验与模型分布

$q(x_{t-1}\mid x_t,x_0)$ 在训练时可解析计算；$p_\theta(x_{t-1}\mid x_t)$ 不使用真实 $x_0$，是生成时实际运行的模型。训练的目标是让后者逼近前者，而不是把二者视为同一个分布。

### 后验方差与模型方差

$\tilde\beta_t$ 由 $\alpha_t,\beta_t,\bar\alpha_{t-1},\bar\alpha_t$ 唯一确定；$\sigma_t^2$ 是模型设计。把 $\sigma_t^2$ 设为 $\tilde\beta_t$ 是一种选择，不是符号改名。

### 等号与正比号

写 $f(y)\propto g(y)$ 时，只能丢掉与待研究变量 $y$ 无关的乘法因子。依赖 $y$ 的项无论看起来多麻烦都不能丢。恢复完整概率密度时还必须补回归一化常数。

### 预测噪声不等于直接做减法

反向更新不是简单的 $x_{t-1}=x_t-\epsilon_\theta$。噪声预测首先通过由噪声日程决定的系数进入 $\mu_\theta$，然后还可能根据 $\sigma_t^2$ 采样随机项。

### 累计系数不是平均值

$\bar\alpha_t$ 是乘积 $\prod_{s=1}^t\alpha_s$，不是 $\alpha_1,\ldots,\alpha_t$ 的算术平均。

### 简化损失不是完整 ELBO 的原样复制

从负 ELBO 到带权噪声 MSE 是参数化后的严格结果；把 $w_t$ 去掉得到 $L_{\mathrm{simple}}$ 是重新加权。工程上常说“DDPM 就是噪声 MSE”时，省略了这层区别。

## 假设、边界与局限

- 前向闭式解和解析后验依赖线性高斯转移；更换破坏过程后不一定保留相同公式。
- 真实一步后验可解析，并不意味着边缘逆分布 $q(x_{t-1}\mid x_t)$ 可直接计算；后者还涉及未知数据分布。
- $x_T\sim\mathcal N(0,I)$ 通常依靠 $\bar\alpha_T$ 足够小来近似成立。
- 噪声方差日程、反向方差选择和时间步数都会影响似然、样本质量与采样成本。
- 本文解释经典离散时间 DDPM，不把后续变体的结论自动套回原始模型。

## 最终总结

DDPM 先规定一个逐步加高斯噪声的前向链。因为每一步都是线性高斯变换，任意 $x_t$ 都能从 $x_0$ 一步采样，同时训练条件下的真实一步反向后验也能用 Bayes 公式和配方法精确求出。

生成时没有真实 $x_0$，因此用网络定义 $p_\theta(x_{t-1}\mid x_t)$。真实后验均值可重写为 $x_t$ 与累计等价噪声 $\epsilon$ 的函数，所以网络可以通过预测噪声来参数化反向均值。

概率论目标仍是最大化数据似然。由于中间轨迹难以积分，DDPM 优化 ELBO；ELBO 的分解产生逐时间步的真实后验与模型分布之间的 KL。固定模型方差后，高斯 KL 的可学习部分化为带权噪声 MSE；经典训练再将其简化为无权的 $L_{\mathrm{simple}}$。

因此最准确的一句话是：DDPM 人为构造一个可解析的数据破坏过程，再用变分学习得到它的概率逆过程；预测噪声是这个概率建模问题经过高斯后验、均值重参数化与目标简化后得到的训练接口。
