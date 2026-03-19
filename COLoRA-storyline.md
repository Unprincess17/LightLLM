# **1. 这个工作准备解决的主要问题？**

在 **Prefill-Decode (PD) 分离** 的在线推理部署中，当一个 decode worker 同时服务：

- **MoE 基座模型**：专家路由动态、访问分布偏斜且具有长尾；
- **Multi-LoRA（多任务微调分支；或者emerging 多租户）**
- **Multi-LoRA 请求流**：多个 adapter 并发活跃，带来更细粒度的参数区分和更高的对象基数；

系统会面临一个比传统 expert-only serving 更棘手的问题：

> 真正需要管理和缓存的对象，不再只是 expert，而是 expert–LoRA 联合单元；这会显著扩大并碎片化有效 working set，使 decode 阶段更早进入 miss 频繁、tail-sensitive 的执行区间。

在这一场景下，有限 GPU 显存不仅需要容纳 KV cache 和基座模型运行态，还需要容纳部分活跃的 expert–LoRA 参数单元。
一旦联合对象的局部性被打散，decode worker 将频繁遭遇参数 miss；若系统仍默认采用阻塞式“换入 GPU 后执行（load-then-run）”，则 PCIe 传输与安装延迟会直接暴露在单请求关键路径上，并显著放大 tail latency。

因此，本文要解决的核心问题不是泛泛的“显存不够”，而是一个更具结构性的执行与代价权衡问题：

> **在 MoE + Multi-LoRA 的 decode 场景下，如何在 expert–LoRA 粒度下管理参数驻留，并在 miss 不可避免时，在“权重迁移 + GPU执行”与“原地 CPU 执行”之间做出代价感知（cost-aware）的选择，从而降低 P99 tail latency。**


## 系统边界

我们聚焦于 **单节点（decode worker）内部** 的异构执行与内存管理，不解决：

- 集群级请求路由，
- 跨节点 KV cache 放置策略，
- 全局调度器的 admission control。

我们假设上层调度器已将请求分配至某个 decode worker；本文解决的是该 worker 内部如何在 **GPU VRAM 与 CPU 内存/算力** 间协同，以降低 decode 阶段的尾延迟并提升资源利用效率。

我们假设 CPU 内存足以容纳完整的 expert–LoRA 参数副本。

# **2. 目前他人工作在此问题上的局限/缺点？**

现有 Multi-LoRA serving 系统（如 MLSys’24 S-LoRA / EuroSys’25 VaLoRA ）在其目标场景下已经展示了很强的参数管理与切换能力，但其设计假设通常更接近：

- 以 LoRA 粒度 为主进行驻留与切换管理；
- 以 GPU 为唯一主要执行位置；
- 将 host memory 主要视为 容量扩展层 / 参数暂存层 / 预取源。

这类设计在纯 Multi-LoRA 或非 MoE 场景下是合理的，但直接迁移到 **MoE + Multi-LoRA @ Decode** 时，会出现两个根本性不匹配。

1. **对象粒度不匹配：expert-only 或 LoRA-only 抽象都会低估真实 working set**

    在 MoE 场景中，请求访问不是静态激活整个 LoRA，也不是均匀访问全部 expert，而是由 router 在 token / layer 级动态选择少量专家。

    一旦引入 Multi-LoRA，请求真正访问的参数对象就不再是“一个 LoRA”或“一个 expert”，而是更细粒度的 expert–LoRA 联合单元。

    如果仍用粗粒度对象做驻留管理，会带来两类问题：

    - 按 LoRA 管理过粗：热点 LoRA 中可能包含大量冷 expert，浪费 VRAM；

    - 按 expert 管理又不完整：忽略 LoRA 维度会错误高估复用，低估缓存压力。

    换言之，expert-only 抽象会系统性低估联合 key space 下的 working-set size 和 miss 风险。

2. **执行范式不匹配：将 miss 一律视为“搬到 GPU 再算”并不总是合理**

    传统 offloading 思路隐含一个默认前提：

    当某个参数单元不在 GPU 上时，合理的处理方式是 先把它搬入 GPU，再执行计算。

    但在 decode 阶段，这一前提并不总成立。原因在于：

    - 请求 batch 小、token 粒度细，难以摊薄固定开销；
    - 长尾 expert–LoRA 单元往往低频、突发、弱复用；
    - 对这类冷对象，阻塞式 promotion 的成本可能大于其后续复用收益；
    - 因而 tail latency 往往更受制于 数据搬运与调度等待，而不只是算子本身。
        
因此，现有“GPU-only execution + host as spill space”的范式，在 MoE + Multi-LoRA decode 下并不充分。

# **3. Case Study 告诉了我们什么？**
为了更具体地理解这一问题，我们构建了一个 trace-driven case study：
将 真实 MoE router trace 与 半真实的 LoRA invocation trace 结合，比较三种访问建模方式：

- B0：expert-only；B0：仅专家模型；
- B1：expert×LoRA（independent）；
- B2：expert×LoRA（correlated）。

这个 case study 并不是为了证明“LoRA 多了肯定更慢”这样显然的结论，而是为了回答：

> **引入 expert–LoRA 联合对象后，系统问题的性质究竟发生了什么变化？**

结果表明，变化并不只是“对象更多了”，而是出现了三个更关键的现象。

1. **联合 keying 会很早将系统推入更碎片化的 locality regime**

相较于 expert-only 建模，expert–LoRA 联合建模的访问分布明显更平、更长尾，热点覆盖率显著下降；
即使 expert 本身仍有偏斜访问，联合对象空间也会因 LoRA 维度的引入而被显著切碎。

这意味着：
> **系统不能再依赖 expert-only 热点直觉来估计驻留压力。**

2. **问题主要集中在 tail requests，而不是所有请求都均匀变差**

进一步分析发现，尾部请求并不是“略微多 miss 一点”，而是会触碰 远多于平均请求 的 cold joint objects。
也就是说，性能恶化并非均匀分布，而是高度集中在一小部分 unlucky requests 上。

这说明系统真正需要处理的，不只是 average locality 下降，而是：

> **tail request 会反复遭遇长尾冷对象，导致 miss 成为结构性而非偶发性事件。**

3. **tail penalty 出现得很早，而且会长期维持在高位平台**

更重要的是，随着 modeled LoRA cardinality 从极小规模开始增长，P99 penalty 会很早出现并迅速抬升；
之后即使继续增加 LoRA 数量，其 tail penalty 往往不是无限制线性增长，而是进入一个 **持续的高位平台**。

这一点的含义是：

> **问题不是“等规模特别大了再考虑”，而是 一旦进入 expert–LoRA fragmentation regime，系统就必须面对结构性的 miss-handling 压力。**

因此，这个 case study 的真正 takeaway 不是“tail latency 很高”，而是：

> **在 expert–LoRA 粒度下，decode worker 的瓶颈已经从单纯的 miss avoidance，转变为如何稳定处理不可避免的长尾 miss。**

这些观察共同指向了一个关键结论：

> **在 expert–LoRA fragmentation regime 下，单纯依赖 miss-avoidance（如更激进缓存或预取）已难以从根本上缓解 tail latency。**

原因在于：

* 联合 key space 膨胀使得有效 working set 超出 GPU 容量，缓存无法覆盖；
* tail request 所触及的对象具有高度不稳定性，难以被预取或预测；
* tail penalty 呈平台化特征，说明问题并不会随资源增加自然消失。

因此，系统瓶颈从“如何避免 miss”，转变为：

> **当 miss 成为结构性事件时，如何以更稳定的方式处理 miss 本身。**

# **4. 我们的主要想法是什么？**

**核心洞察：并非所有 expert-LoRA 调用都值得走同一条执行路径**

case study 表明，在 decode worker 中，一旦 expert–LoRA 联合对象的长尾访问成为结构性现象，系统就不能再把所有 miss 都视为同一种事件统一处理。

对于 **高频、可复用的热点 expert–LoRA** 单元，保持 GPU 驻留并走 GPU 路径仍然是最优选择；

但对于 **低频、突发、弱复用的冷单元**，若一律采用阻塞式“换入 GPU 后执行”，请求 tail latency 往往会被绑在最慢的 promotion 路径上。
对这类对象而言，可以将 miss-handling 视为一个延迟权衡问题：

* **GPU 路径**：PCIe 传输 + GPU 执行（但需阻塞等待权重到达）
* **CPU 路径**：直接在 host 侧执行（无迁移，但计算能力较弱）

在 decode 场景中，由于 batch 小、调用碎片化，PCIe 传输与调度延迟往往难以摊薄。对于低频冷对象，其复用间隔通常较长（i.e., large reuse distance），导致一次权重迁移的成本难以在后续调用中被有效摊薄（amortize），从而使得：

> **“搬入 GPU 再执行”在延迟上不一定优于“直接在 CPU 执行”。**

因此，本文的关键判断是：

> **host memory 不应只被视为更慢的显存扩展层，还应被视为一个可参与 miss-time 计算的辅助执行层。**

这意味着系统设计的重点不再只是“哪些对象该留在 GPU 上”，还包括：

> **当长尾 miss 不可避免时，系统应该如何处理这些 miss。**


# **5. COLoRA 的总体思路是什么？**

基于上述观察，COLoRA 在 decode worker 中采用一种 **面向 expert–LoRA 联合对象的异构双路径执行策略**：

- **GPU 路径**：服务热点、可复用的 expert–LoRA 单元；

- **CPU 路径**：作为 cache miss / 长尾冷单元的 fallback 执行路径，避免请求在 miss 时一律阻塞等待权重换入。

配合两类关键机制：

- **非对称内存池与细粒度驻留管理**：GPU 维护容量受限的 hot expert–LoRA cache，CPU 内存维护 full replica / cold storage；

- **轻量级 CPU 执行与通信-计算重叠**：尽量将 CPU fallback 的固定开销压低，并减少其对 GPU 主时间线的阻塞。

COLoRA 的目标不是消灭 PCIe 代价，也不是让 CPU 替代 GPU 做主计算；
它要做的是：

> **在 expert–LoRA fragmentation 下，用更稳定、可控的 fallback 路径替代高抖动的阻塞式 promotion，从而改善 decode 阶段的 tail latency。**

此外，需要指出的是，COLoRA 并不在所有场景下引入额外复杂性：当访问局部性较高或 GPU 容量足以覆盖活跃对象时，系统将自然退化为 GPU-only 执行路径，从而避免不必要的异构执行与调度开销。

# **6. 这一路径面临哪些技术挑战？**

## 挑战一：联合 key space 膨胀下的细粒度驻留管理

在 MoE 路由和 Multi-LoRA 并发共同作用下，访问热点不再只呈现 expert 偏斜，而是表现为 **expert × LoRA 的联合偏斜**。

这会带来两个直接后果：

- 粗粒度管理（按 LoRA 或更大块）会显著浪费 VRAM；

- 细粒度管理（按 expert–LoRA 单元）虽然更精确，但也会让 miss 更频繁、更碎片化。

因此，系统必须在有限 GPU 空间下识别并维持一个足够小但足够有效的热点集合，同时避免频繁 churn 和无效 promotion。

**核心需求**：需要一种在 expert–LoRA 粒度下进行驻留管理的机制，既能利用细粒度带来的精确性，又不会让系统因为对象数爆炸而陷入过度 thrashing。

## 挑战二：如何把 miss-handling 从“阻塞式换入”变成“可控 fallback”

case study 的关键启示是：
在 expert–LoRA fragmentation regime 下，miss 并不是偶发异常，而是结构性存在。

因此，系统不能把 miss 一律等同于 “load-then-run”。
尤其在 decode 路径上，某些冷对象的未来复用很弱，阻塞式 promotion 未必划算。

但 CPU fallback 也不是天然高效的：

- decode 阶段 batch 小、调用碎片化；

- CPU 算子容易被框架 dispatch / tensor orchestration 固定开销吞掉；

- 若 fallback 成本本身过高，只是把“传输延迟”换成了“CPU 调度延迟”。

**核心需求**：需要一个低固定开销的 CPU fallback 执行路径，使其真正成为一种比阻塞式 promotion 更稳定的 miss-time 响应方式。

## 挑战三：双执行路径下的同步、重叠与尾部稳定性

一旦同一层内的部分 expert–LoRA 单元在 GPU 执行、部分在 CPU 执行，系统就会引入额外的：

- 激活值在 CPU/GPU 间传输，

- 异构路径的结果合并与同步，

- CPU 与 GPU 时间线的耦合，

- CPU fallback 队列自身的抖动。

如果处理不当，GPU 可能因为等待 CPU 返回而空转，或者 CPU fallback 本身成为新的 tail source，从而抵消设计收益。

**核心需求：**：需要在可行的数据依赖边界内最大化通信与计算重叠，并控制 CPU 路径排队与同步成本，使 dual-path 真正改善 tail latency，而不是引入新的瓶颈。

# **7. 我们工作的主要方法是什么，以及三个主要的创新点？**

## 方法概述

COLoRA 是一个面向 **decode worker** 的异构推理执行引擎。它在 **expert–LoRA 联合粒度** 下进行参数管理，并通过：

* **非对称缓存（GPU hot cache + CPU full replica）**
* **CPU fallback 执行路径**

在运行时动态选择 GPU 或 CPU 执行，以降低 MoE + Multi-LoRA 场景下长尾访问导致的阻塞式权重换入开销。

在执行层面，CPU fallback 仅负责 **LoRA residual 分支（低秩增量）** 的计算，而 base expert 主路径始终保持在 GPU 上执行，以避免对主干精度与吞吐造成影响。

## **三个主要创新点：**

### 创新点 1：LoRA-Expert 交叉粒度的非对称内存池与非阻塞 miss 处理

*(Memory policy，核心是“让 miss 不再阻塞”)*

我们提出一种面向 MoE LoRA 场景的 **LoRA-Expert 交叉粒度内存管理机制**，并将 cache miss 从“阻塞事件”转化为“可调度事件”。

- 在 GPU 侧维护容量受限的 **hot LoRA-Expert cache**，以 `(LoRA_a, Expert_b)` 作为最小驻留单元；
- 在 CPU 内存中维护对应单元的 **全量只读副本**，作为冷数据层与 fallback 执行来源；
- 当发生 GPU cache miss 时，系统不再采用阻塞式 load-then-run，而是基于运行时状态进行选择：

  - 对具有热点回升趋势的单元执行 **异步提升（promotion）** 至 GPU cache；；
  - 对当前请求直接走 **CPU fallback 执行路径**。

该机制的关键在于：

> **cache miss 不再必然转化为请求阻塞，而是被转化为异构执行路径的调度选择。**

从而将 miss 的代价从高抖动的权重迁移延迟，转化为更可控的计算与传输开销，为 decode 场景下的尾延迟优化提供基础。

### 创新点 2：面向 decode 长尾调用模式的低开销 CPU fallback 执行机制

*(Compute viability，核心是“让 fallback 变得可用”)*

为使 CPU fallback 路径在 decode 场景下真正成为可行选择，我们设计了一条 **面向小批量、碎片化 expert 调用的低固定开销执行路径**。

具体而言，我们通过轻量化算子实现（如基于 AVX 的微内核）与更精简的调度封装，系统性降低：

- 框架级 dispatch 开销，
- 小张量组织与调度成本，
- 微型 GEMV/GEMM 的固定启动开销。

该设计的目标并非提升 CPU 的峰值计算能力，而是：

> **压缩 CPU fallback 的固定成本，使其在低频、长尾、时延敏感的调用中具备稳定且可接受的执行代价。**

这一点是整个系统成立的关键前提：

> 若 fallback 路径本身开销不可控，则“非阻塞 miss”将退化为另一种形式的尾延迟来源。

因此，该机制本质上提供了一个 **可用（viable）且稳定的 fallback 执行基础**。

### 创新点 3：基于时间局部性的投机发射与延迟绑定流水线 / Temporal-Locality-Guided Speculative Dispatch with Late Binding for Heterogeneous Overlap

*(Scheduling policy，核心是“让 fallback 被重叠”)*

在具备非阻塞 miss 与可行 fallback 的基础上，我们进一步提出一种 **基于时间局部性的投机调度机制**，用于消除 CPU/GPU 异构路径之间的同步等待。

核心思想是：

> **利用 decode 过程中的时间局部性，提前启动 CPU fallback 计算，并在需要时再决定是否使用其结果（late binding）。**

具体而言：

- 基于真实 trace 观察到的 **MoE 路由时间局部性及其层级分布特征**，系统对不同层采用自适应策略：

  - 在浅层与深层等高稳定区，启用 **投机发射（speculative dispatch）**；
  - 在中间层等低置信区，关闭投机，回退至常规执行；
- 在解码步 *t* 处理第 *L* 层时：

  - 利用前一时刻的路由状态进行预测；
  - 将潜在需要的 LoRA fallback 任务提前派发至 CPU 并启动计算；
- 在 GPU 执行至该层同步点时，通过 **延迟绑定（late binding）** 决定：

  - 若预测命中，则合并 CPU 结果；
  - 若预测失败，则直接丢弃。

为控制精度风险，该机制仅作用于 **LoRA residual path**，而不影响 base FFN 主路径的精确计算。

该机制带来的关键变化是：

> 将 CPU fallback 从“被动响应 miss”转变为“可提前调度的异步执行单元”，并将其执行时间隐藏在 GPU 主路径之下。

从而显著减少异构路径之间的同步等待，提高流水线重叠程度，并进一步优化尾延迟表现。

# **8. 我们最重要的创新是什么？我们工作主要的局限性是什么？**

## 最重要创新

COLoRA 的关键贡献不在于简单引入 CPU 参与计算，而在于提出了一种 **面向长尾 miss 的执行范式转变（miss-handling paradigm shift）**：

> **将 cache miss 从“必须通过参数迁移解决的问题”，重构为“可以通过异构执行路径吸收的运行时事件”。**

具体而言，在 decode 场景下，传统系统将 miss 统一转化为阻塞式“load-then-run”，使 PCIe 传输与调度延迟直接暴露在关键路径上；而 COLoRA 通过：

* GPU hot cache（服务高复用对象）
* CPU full replica（提供无迁移执行能力）
* CPU fallback execution（处理低复用长尾调用）

构建了一种 **双路径执行模型（dual-path execution model）**，使得：

> cache miss 不再必然转化为阻塞等待，而是可以被转化为 **可调度、可重叠的计算路径**。

这一转变将系统优化重点从“尽量避免 miss”转向“稳定处理不可避免的 miss”，从而在 MoE + Multi-LoRA 的长尾访问场景下显著改善 tail latency。

## 主要的局限性：

## 局限 1：收益依赖于路由长尾程度与 CPU fallback 负载占比

COLoRA 的性能优势依赖于如下条件成立：

* 热点 expert–LoRA 单元能够稳定驻留 GPU；
* 冷对象访问呈现低频、分散特征；
* CPU fallback 调用比例处于可控范围内。

当请求模式发生变化，例如：

* 大规模突发冷路由（burst of cold accesses），或
* fallback 比例持续升高，

CPU 侧可能形成排队，且其执行与传输开销无法被 GPU 主路径有效重叠，从而削弱 tail latency 改善效果，甚至引入新的尾部等待。

> 本质上，该系统依赖于“热-冷分离 + fallback 稀疏”的结构性假设。

## 局限 2：CPU 路径优化存在硬件依赖性

CPU fallback 的性能高度依赖于底层硬件与系统配置，包括：

* CPU 微架构（向量宽度、缓存层级）
* NUMA 拓扑
* 线程调度与绑定策略

当前实现主要依赖离线 profiling 确定参数（如分块大小、线程布局），运行时自适应能力有限。因此，在不同硬件平台或资源竞争环境下，fallback 路径性能可能出现显著波动。

## 局限 3：仍然受限于异构通信与系统拓扑

COLoRA 减少的是**阻塞式权重迁移带来的延迟暴露**，但并未消除异构通信成本。

在以下场景中：

* hidden dimension 较大（激活体积高）
* PCIe / NUMA 拓扑复杂
* 与其他通信流量（如 KV cache 传输）竞争带宽

CPU 与 GPU 间的激活传输仍可能成为瓶颈，从而限制整体收益。

# **9. 我们工作主要的比较对象是什么？主要的衡量指标是什么？**

## **评估设置概述**

我们的主要评估场景为 **Multi-LoRA @ Decode**，即多个 LoRA adapter 并发活跃，并在 decode 阶段产生细粒度 expert–LoRA 参数访问的典型部署模式。这一设置对应实际多租户推理服务中最容易出现 **working set 膨胀与访问碎片化** 的场景。

同时，为了验证方法的适用范围与鲁棒性，我们额外考虑以下两类扩展设置：

* **Single-LoRA 场景（对照）**：用于评估在无多租户干扰、参数复用更强时，COLoRA 是否仍能保持稳定性能或自然退化为 GPU-only 执行；
* **Prefill-Decode 未分离场景（扩展分析）**：用于分析在 unified serving 下，prefill 负载对 CPU fallback 路径与异构调度的潜在影响（通过受控负载注入与资源竞争模拟进行分析）。

> 这些扩展设置的目标不是改变问题定义，而是验证 COLoRA 的设计是否依赖于特定部署假设。

---

## **比较对象：**

为了全面评估 COLoRA 的有效性，我们设计了以下三类基线：

1. **Optimized GPU-only Multi-LoRA/MoE baseline（强基线）**

   代表当前主流部署范式，在 GPU 上执行所有计算，并配备：

   * 合理的缓存策略（LoRA / expert-aware）
   * 预取与调度优化

   该基线用于衡量：**在不引入异构执行的前提下，系统所能达到的最优性能。**


2. **Fine-grained GPU-only baseline（粒度增强基线）**

   在 GPU-only 框架下引入 expert–LoRA 粒度的缓存与调度策略，用于验证：

   > **仅依赖更细粒度的驻留管理，是否足以缓解 fragmentation 带来的 tail latency 问题。**


3. **COLoRA 消融实验（机制拆解）**

   用于隔离各组件贡献：

   * w/o CPU fallback（全部 miss 采用阻塞换入）
   * w/o overlap（禁用异构重叠）
   * w/o optimized CPU kernels（使用通用框架路径）

---

## **扩展对照实验（适用性验证）**

为了验证 COLoRA 的设计是否依赖于 multi-LoRA fragmentation，我们进一步设计以下对照实验：

### - Single-LoRA 场景

仅启用单一 LoRA adapter，使参数访问具有更强的复用性与稳定性。

该实验用于验证：

* CPU fallback 触发比例是否显著下降
* 系统是否自然退化为 GPU-only 执行路径
* 是否引入额外调度或通信开销

> 该结果用于说明：COLoRA 在低 fragmentation 场景下不会带来负面影响。

### - Unified Serving 场景（Prefill-Decode 未分离）

通过引入受控的 prefill 负载，与 decode 请求共享 GPU 与 CPU 资源，用于分析：

* CPU fallback 与 prefill 计算之间的资源竞争
* 异构路径重叠是否被打破
* tail latency 是否受到额外扰动

该实验主要用于定性分析 COLoRA 在非理想部署条件下的表现。


## **衡量指标**

我们使用以下指标评估系统性能：

---

### - 延迟指标（核心）

* **TPOT（平均 / P90 / P99）**
  衡量 decode 阶段每 token 生成延迟，是本文的核心指标

---

### - 吞吐指标

* **Throughput（req/s 或 tok/s）**
  衡量系统整体处理能力

---

### - 端到端指标

* **TTFT（Time-To-First-Token）**
  用于验证 decode 优化是否对 prefill 或整体调度产生副作用

---

### - miss 行为分析

* **GPU cache miss rate**
* **miss handling breakdown（阻塞换入 vs CPU fallback）**

用于回答：

> tail latency 的变化是否来自 miss-handling 机制本身

---

### - CPU fallback 细粒度分析

* **CPU fallback queueing time**
* **CPU execution time**
* **activation transfer time（CPU↔GPU）**

用于解释：

> fallback 路径是否成为新的瓶颈

---

### - 资源利用率

* **GPU utilization / SM occupancy**
* **CPU utilization**

用于评估：

> 异构执行是否提升整体资源利用效率

---

### - 内存开销

* **Host memory footprint（CPU 全量副本）**

用于量化：

> COLoRA 引入的额外资源成本

---

## **评估目标总结**

上述指标共同用于回答两个核心问题：

1. **COLoRA 是否显著降低 decode 阶段的 tail latency？**
2. **这种改善是否来源于 miss-handling 范式转变，而非其他系统因素？**