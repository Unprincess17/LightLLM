# **1. 这个工作准备解决的主要问题？**

在 **Prefill-Decode (PD) 分离** 的在线推理部署中，当模型同时具备：

- **MoE 架构**（动态专家路由，参数规模大、访问长尾），以及
- **Multi-LoRA（多任务微调分支；或者emerging 多租户）**

时，Decode 节点会面临一个突出的系统瓶颈：

**有限 GPU 显存需要同时容纳 KV cache、基座模型运行态以及部分活跃 LoRA/MoE 参数，导致显存竞争加剧；一旦参数换入换出频繁，PCIe 传输开销会显著拉高尾延迟。**

## 系统边界

我们聚焦于 **单节点（decode worker）内部** 的异构执行与内存管理，不解决：

- 集群级请求路由，
- 跨节点 KV cache 放置策略，
- 全局调度器的 admission control。

我们假设上层调度器已将请求分配至某个 decode worker；本文解决的是该 worker 内部如何在 **GPU VRAM 与 CPU 内存/算力** 间协同，以降低 decode 阶段的尾延迟并提升资源利用效率。

# **2. 目前他人工作在此问题上的局限/缺点？**

现有 Multi-LoRA serving 系统（如 MLSys’24 S-LoRA / EuroSys’25 VaLoRA ）在其目标场景下表现优秀，但其设计重点通常是：

- LoRA 粒度的参数管理与切换；
- 以 GPU 计算为主；
- Host memory 主要作为容量扩展层（参数暂存/预取源）。

当上述设计直接迁移到 **MoE + Multi-LoRA @ Decode** 场景时，会出现两类不匹配：

1. **粒度不匹配（LoRA-level vs. Expert-level）**
    
    MoE 路由使参数访问呈现更强的动态稀疏性与长尾性。若仍以粗粒度管理参数驻留，可能导致：
    
    - 热 LoRA 中包含冷 Expert，造成显存利用率下降；
    - 为避免浪费而细化到按 Expert 加载时，又引入频繁 cache miss 与传输抖动。
2. **执行范式不匹配（“仅搬运权重到 GPU”）**
    
    对于 decode 阶段中部分低并发、碎片化的专家调用，继续坚持“权重搬到 GPU 再算”，可能使时延更受制于：
    
    - 权重传输与调度开销，
    - 而非实际计算本身。
        
        这类路径在 tail latency 上尤其敏感。
        

# **3. 我们的主要想法/思路是什么？对此，我们面临什么技术挑战？**

## 核心洞察 - 异构双执行路径？

在 decode 节点中，并非所有 expert 调用都值得采用同一种执行路径。

对于高频访问的 expert，GPU 执行仍然更合适；

而对于低频、突发、长尾 expert，**将权重搬入 GPU 的代价**在某些情况下可能高于**直接在 CPU 侧完成该 expert 计算并回传激活值**的代价。

因此，与其把 host memory 仅视作“更慢的显存扩展”，不如将其同时视作：

- **参数驻留层**（full replica / cold storage）和
- **可参与计算的执行层**（CPU fallback executor）。

## 总体思路

COLoRA 在 decode worker 内采用一种 **异构双路径执行策略**：

- **GPU 路径**：服务热点 expert（低延迟、高吞吐）
- **CPU 路径**：作为 cache miss / 长尾 expert 的 fallback 执行路径（避免阻塞式权重换入）

配合：

- expert 级别的非对称缓存管理（GPU hot cache + CPU full replica），以及
- 轻量化 CPU 算子 + 通信/计算重叠机制，

目标不是“消灭”PCIe 成本，而是**在长尾路由场景下，用更稳定的 fallback 路径替代高抖动的阻塞式权重换入，从而改善 tail latency。**

## 技术挑战：

### 挑战一：二维交叉热点下的缓存粒度与 miss 放大效应

在 MoE 路由与 Multi-LoRA 并发共同作用下，参数访问热点分布呈现 `LoRA × Expert` 的交叉偏斜特征。并且长尾 expert 的到达具有较高不确定性，预取可降低平均 miss 但难以消除尾部 miss。

- 粗粒度（例如按 LoRA 或更大块）缓存管理容易造成 VRAM 浪费；
- 细粒度（按 Expert）虽更精确，但 miss 更频繁；
- 若 miss 处理采用阻塞式换入（load-then-run），decode 阶段的单 token 延迟容易出现长尾放大。

**核心需求**：需要一种既能细粒度管理 LoRA expert，又不把 miss 直接转化为阻塞等待的机制。

### 挑战二：CPU fallback 路径的微小计算效率问题

将冷专家计算下沉到 CPU 并不自动带来收益。decode 场景常见的问题是：

- batch size 小，
- 调用碎片化，
- 路由动态变化，
- 框架级 dispatch / kernel launch / tensor orchestration 开销占比高。

**核心需求**：CPU 路径必须有足够低的固定开销，否则 fallback 只会把“传输延迟”换成“框架调度延迟”。

### 挑战三：异构双路径引入的同步与流水线气泡

当同一层内部分 expert 在 GPU、部分在 CPU 执行时，会引入额外的：

- 激活值传输（CPU<->GPU），
- 结果合并同步点，
- CPU/GPU 时间线耦合。

如果处理不当，GPU 可能因等待 CPU 返回而空转，抵消 fallback 带来的收益。

**核心需求**：需要在可行的数据依赖边界内尽量重叠通信与计算，并控制 CPU 路径的排队与抖动。

# **4. 我们工作的主要方法是什么，以及三个主要的创新点？**

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