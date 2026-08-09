<!--
  本文档为 TensorRT-LLM 官方 Overlap Scheduler 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/overlap-scheduler.md
-->

# Overlap Scheduler（重叠调度器）

为了最大化 GPU 利用率，调度器将 CPU 任务（例如检查采样停止条件、更新响应、调度下一批）与 GPU 计算重叠执行。

> 💡 **AI Infra 视角**：这是"隐藏 CPU 延迟"的经典做法，[架构总览](../developer-guide/overview.md)中的 Overlap Scheduler 小节有详细讲解。核心直觉：GPU 和 CPU 是两条独立的流水线——CPU 在等 GPU 算完时闲着，GPU 在等 CPU 准备数据时也闲着。重叠调度就是把两边的空闲时间互相填上：**先把 GPU 下一步的活全部 launch 出去（异步），趁 GPU 在跑，CPU 回头处理上一步的采样/停止判断**。

## 工作原理

在第 *n* 步，系统直接启动第 *n+1* 步的 GPU 计算，而不等待第 *n* 步的 CPU 任务（如停止条件检查）完成。这样：

- CPU 工作（第 *n* 步）和 GPU 计算（第 *n+1* 步）可以并发运行。
- GPU 空闲时间减少，占用率（occupancy）提高。

这个并发执行流水线在 `PyExecutor` 的逻辑中体现：

```python
# 调度并启动当前步骤 (n) 的 GPU 工作
scheduled_batch, _, _ = self._schedule()
batch_outputs = self._forward_step(scheduled_batch, previous_tensors_device)
sample_state = self._sample_async(scheduled_batch, batch_outputs)

# GPU 忙碌期间，处理上一步 (n-1) 的 CPU 侧结果
if self.previous_batch is not None:
    self._process_previous_batch()
```

## 权衡（Tradeoff）

这个优化引入了额外的一个解码步骤（流水线初始填充），但显著提升了吞吐。

> 💡 **AI Infra 视角**：代价是"多算一步"：流水线刚启动时，第 0 步的 CPU 活没有上一步可以掩盖，要等；相当于每个请求的延迟多了一步的时间。但换来的是持续运行中 GPU 不再空等——**以少量延迟换大量吞吐**，对服务端场景几乎总是划算的（这也是它默认开启的原因）。理解这种"软件流水线"思想，对理解 CUDA Graph、双缓冲（double buffering）等优化都有帮助。

## 使用方法

默认启用。要禁用，在配置中设置 `disable_overlap_scheduler=True`。

## 参考资料

- [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/abs/2408.12757)
- https://lmsys.org/blog/2024-12-04-sglang-v0-4/#zero-overhead-batch-scheduler
