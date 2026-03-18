# **1. 这个工作准备解决的主要问题？**

在 **Prefill-Decode (PD) 分离** 的在线推理部署中，当一个 decode worker 同时服务：

- **MoE 基座模型**：专家路由动态、访问分布偏斜且具有长尾；
- **Multi-LoRA（多任务微调分支；或者emerging 多租户）**
- **Multi-LoRA 请求流**：多个 adapter 并发活跃，带来更细粒度的参数区分和更高的对象基数；

系统会面临一个比传统 expert-only serving 更棘手的问题：

> 真正需要管理和缓存的对象，不再只是 expert，而是 expert–LoRA 联合单元；这会显著扩大并碎片化有效 working set，使 decode 阶段更早进入 miss 频繁、tail-sensitive 的执行区间。

在这一场景下，有限 GPU 显存不仅需要容纳 KV cache 和基座模型运行态，还需要容纳部分活跃的 expert–LoRA 参数单元。
一旦联合对象的局部性被打散，decode worker 将频繁遭遇参数 miss；若系统仍默认采用阻塞式“换入 GPU 后执行（load-then-run）”，则 PCIe 传输与安装延迟会直接暴露在单请求关键路径上，并显著放大 tail latency。

因此，本文要解决的核心问题不是泛泛的“显存不够”，而是：

> **在 MoE + Multi-LoRA 的 decode 场景下，如何在 expert–LoRA 粒度下管理参数驻留，并在 miss 不可避免时，以更稳定的方式处理长尾访问，从而降低 P99 tail latency。**


## 系统边界

我们聚焦于 **单节点（decode worker）内部** 的异构执行与内存管理，不解决：

- 集群级请求路由，
- 跨节点 KV cache 放置策略，
- 全局调度器的 admission control。

我们假设上层调度器已将请求分配至某个 decode worker；本文解决的是该 worker 内部如何在 **GPU VRAM 与 CPU 内存/算力** 间协同，以降低 decode 阶段的尾延迟并提升资源利用效率。

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

这也直接引出了本文的设计动机。

# **4. 我们的主要想法是什么？**

**核心洞察：并非所有 expert-LoRA 调用都值得走同一条执行路径**

case study 表明，在 decode worker 中，一旦 expert–LoRA 联合对象的长尾访问成为结构性现象，系统就不能再把所有 miss 都视为同一种事件统一处理。

对于 **高频、可复用的热点 expert–LoRA** 单元，保持 GPU 驻留并走 GPU 路径仍然是最优选择；

但对于 **低频、突发、弱复用的冷单元**，若一律采用阻塞式“换入 GPU 后执行”，请求 tail latency 往往会被绑在最慢的 promotion 路径上。
对这类对象而言，**将权重搬入 GPU 的代价**在某些情况下可能高于**直接在 CPU 上完成该单元计算并回传激活值的代价。**

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

COLoRA 是一个面向 **decode worker** 的异构推理执行引擎。它通过 **expert 级非对称缓存** 和 **CPU fallback 执行路径**，在 expert 粒度上动态选择 GPU 或 CPU 执行，以降低 MoE-LoRA 长尾路由导致的阻塞式权重换入开销；同时通过轻量化 CPU 算子与异构重叠执行，减少 fallback 带来的同步代价。

## **三个主要创新点：**

### 创新点 1：LoRA-Expert 交叉粒度的非对称内存池与非阻塞 miss 处理

- 在 GPU 侧维护容量受限的 **hot LoRA-Expert cache**，以 `(LoRA_a, Expert_b)` 交叉单元作为缓存与驻留管理的最小粒度；
- 在 CPU 内存中维护对应 **LoRA-Expert 单元权重的全量只读副本**，作为冷数据驻留层与 fallback 执行的数据来源；
- 当某个 `(LoRA_a, Expert_b)` 单元发生 GPU cache miss 时，系统不再默认采用阻塞式“换入后执行（load-then-run）”，而是可按运行时状态选择：
    - 对具有热点回升趋势的单元进行异步提升（promotion）至 GPU cache；
    - 对当前请求直接走 CPU fallback 路径完成该单元对应的 LoRA 计算，避免请求在 miss 上阻塞等待权重换入。

该机制使 **cache miss 不再必然转化为请求阻塞**，从而将 miss 代价从高抖动的阻塞式权重迁移，转化为更可控的异构执行路径，并为改善 decode 阶段的尾延迟提供空间。

### 创新点 2：面向 decode 稀疏调用模式的低开销 CPU fallback 算子

**(计算路径创新，重点是“降低固定开销”)**

我们为 CPU fallback 路径实现了针对小批量/碎片化 expert 调用优化的轻量算子路径（基于 AVX 的微内核与更轻的调度封装），目标是降低：

- 框架 dispatch 开销，
- 小张量组织开销，
- 微型 GEMV/GEMM 的固定成本。

这并不试图在绝对 FLOPs 上超过 GPU，而是使 CPU fallback 在 **长尾、低频、时延敏感** 的调用上成为可用且稳定的替代路径。

### 创新点 3：异构双路径下的重叠执行与流水线化调度

**(调度创新，重点是“减少等待”)**

我们将 CPU fallback 的计算与数据传输封装为可异步调度的执行单元，并在满足数据依赖的前提下，尽可能与 GPU 侧主路径执行重叠。与此同时，在 CPU 内部对多个专家任务进行轻量流水化，以降低 cache miss 和调度抖动造成的尾部放大。

该设计的目标是：

- **减少**异构路径带来的同步等待，
- 提高双路径并行时的时间线稳定性。

# **5. 我们最重要的创新是什么？我们工作主要的局限性是什么？**

## 最重要创新

COLoRA 的关键贡献不是“把 CPU 当成第二个 GPU”，而是提出并实现了一种 **面向 MoE 长尾路由的非阻塞 miss 处理范式**：

> 在 decode 场景下，系统不再将 cache miss 一律转化为“参数换入等待”，而是通过“GPU hot cache + CPU full replica + CPU fallback execution”的异构双路径机制，将 miss 的代价从高抖动的阻塞式权重传输，转化为更可控的 fallback 计算与激活传输路径，并通过重叠调度进一步降低对 tail latency 的影响。
> 

# 主要的局限性：

## 局限 1：收益依赖于路由长尾程度与 CPU fallback 负载占比

COLoRA 的优势主要体现在：

- 热点专家可稳定驻留 GPU，
- 冷门专家访问相对稀疏、
- CPU fallback 不形成持续排队。

当出现大规模突发冷路由、或 CPU fallback 比例持续升高时，CPU 侧排队与传输开销可能突破可重叠窗口，导致 tail latency 收益下降，甚至引入额外等待。

## 局限 2：CPU 路径优化存在硬件依赖性

CPU fallback 算子的最佳配置（如向量化策略、任务分组大小、线程绑定方式）与 CPU 微架构、NUMA 拓扑、缓存层级密切相关。当前实现主要通过离线 profiling 选取参数，运行时自适应能力有限。

## 局限 3：仍然受限于异构通信与系统拓扑

COLoRA 减少的是**部分权重传输造成的阻塞**，并不消除异构通信成本。对于高 hidden dimension、复杂 PCIe/NUMA 拓扑，或与其他通信流量（例如系统内其他 DMA/KV 相关传输）争用链路的场景，激活值往返仍可能成为重要限制因素。

# **6. 我们工作主要的比较对象是什么？主要的衡量指标是什么？**

## **比较对象：**

（multi-lora为主, single-lora辅助；pd不分离？）

为了隔离各模块贡献，评估应包含三类基线：

1. **GPU-only Multi-LoRA/MoE Serving baseline（强基线）**
    
    代表主流“参数尽量在 GPU 执行”的范式，并配备合理的预取/缓存策略。
    
2. **Expert-granular GPU-only baseline（细粒度增强基线）**
    
    用于验证：仅靠 expert 级缓存管理是否足以解决 decode 的尾延迟问题。
    
3. **COLoRA 系列消融（机制拆解）**
    - w/o CPU fallback（miss 改为阻塞换入）
    - w/o overlap（同步执行）
    - w/o optimized CPU kernels（使用通用框架路径）

## **衡量指标：**

- **TPOT（平均 / P90 / P99）**：核心指标，反映 decode 时延与长尾
- **Throughput（req/s 或 tok/s）**
- **TTFT**：验证 decode 优化是否引入 prefill/调度副作用（若端到端评估）
- **GPU cache miss rate / miss handling breakdown**（阻塞换入 vs CPU fallback）
- **CPU fallback queueing time / execution time / transfer time**（把机制解释清楚）
- **GPU/CPU 利用率与资源占用**
- **Host memory footprint**（因为用了 CPU 全量副本）

# **7. Case Study：Real Router + LoRA Invocation Trace**

## 目标

用真实 MoE 路由与 LoRA 调用轨迹，构造 `Expert × LoRA` 的真实组合分布，验证：

- `Expert × LoRA` 的交叉稀疏会显著放大 GPU cache miss
- miss 放大进一步加剧 decode 尾延迟（P90/P99/TPOT）
- 仅靠 expert 级缓存管理不足以稳定长尾

## 数据来源与采集

**Real router trace（MoE 路由）**

- 选择一个真实 MoE 模型（如 Mixtral / Qwen-MoE / Switch 类），在 decode 阶段记录路由结果
- 每个 token、每层记录 top-k expert id 与 gate weight
- 需要保留请求 id、时间戳、层号、batch size、token 位置等信息，用于复现调度与并发

**Real / Synthetic LoRA trace（LoRA 调用）**

- 若有线上日志：记录 LoRA id、请求到达时间、token 数、并发会话长度
- 若无真实日志：构造合成 trace
- 合成 trace 建议满足：
- LoRA 热度服从 Zipf（长尾明显）
- 具有 burst / session 行为（尾延迟放大更明显）

## Trace 事件格式（建议）

```text
router_event:
  t, req_id, layer, token_pos, topk_experts[], topk_weights[], batch_size

lora_event:
  t, req_id, lora_id, input_len, output_len
```

## 组合方法（构造 Expert × LoRA）

**Join 规则**

- 以 `req_id` 为键，将 LoRA 调用信息与 router 事件关联
- 产生 `lora_id + expert_id` 的组合事件
- 对同一 req 的所有 layer / token 形成真实的专家调用序列

**时间轴一致性**

- 保留原始时间戳，复现 batcher 形成的并发
- 不做强行对齐，让 tail burst 自然出现

**两类组合场景**

- Independent：LoRA 与 router 独立组合，用于“平均”行为对照
- Correlated：人为绑定“某些 LoRA 更偏某些 expert”的相关性，用于 worst-case stress

## 实验设计

**对照组**

1. MoE-only（单 LoRA）：仅 router trace，验证专家长尾但无 LoRA 维度
2. Multi-LoRA-only（dense 模型）：仅 LoRA trace，验证 LoRA 切换但无专家维度
3. MoE × Multi-LoRA（真实组合）：核心 case study

**系统配置控制**

- 固定 GPU cache 容量
- 固定预取策略与替换策略
- 统一 batcher 与 decode 线程模型

## 关键观测指标

- GPU cache miss rate（按 Expert / LoRA-Expert）
- miss 处理路径占比：阻塞换入 vs CPU fallback
- PCIe 传输字节与时延
- TPOT P50/P90/P99（核心 tail latency）
- LoRA-Expert 热度分布与碎片化程度（访问频率直方图）

## 预期结论（Case Study 的价值）

- `MoE × Multi-LoRA` 的交叉稀疏度明显高于任何单维度变化
- Expert 级缓存管理虽降低平均 miss，但对尾部 miss 无法稳定消除
- 需要非阻塞 miss 处理与 fallback 执行来压制 tail latency 放大

## Checklist

- [ ] real router
- [ ] real / synthetic lora trace
- [ ] combine into a case study
