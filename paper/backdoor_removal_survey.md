# 后门消除/缓解型防御调研报告

## 1. 调研目标与防御假设

本报告面向下一步 CT-BA/JSCC 后门消除研究。当前防御流程假设如下：

- 已经完成第一步：检测模型存在后门。
- 防御者只能获得后门模型和少量/适量干净数据。
- 防御者没有干净模型、没有毒化样本、没有训练阶段控制权，也不知道触发器。
- 目标不是运行时拦截触发样本，而是修复模型，使其在触发条件下不再输出目标图像，同时尽量保持主任务重建质量。

该设定与传统图像分类后门修复中的“post-training model repair with clean data”最接近，但 CT-BA 的触发器位于信道/latent 统计域，输出任务是图像重建而不是分类。因此，现有方法需要重新解释和改造。

## 2. 检索策略

检索日期：2026-06-13。

主要来源：

- arXiv
- Google Scholar 方向检索词
- Semantic Scholar/DBLP/论文页面作为补充元数据来源
- 本地论文：`paper/2503.23866v3.pdf` 与 `paper/2503.23866v3.md`

关键词组合：

- `backdoor removal clean data backdoored model`
- `backdoor model repair clean data no trigger`
- `fine-pruning backdoor defense`
- `neural attention distillation backdoor defense`
- `adversarial neuron pruning backdoor defense`
- `implicit backdoor adversarial unlearning`
- `reconstructive neuron pruning backdoor defense`
- `fine-tuning backdoor defense sharpness-aware minimization`
- `semantic communication backdoor defense`
- `channel-triggered backdoor attack semantic communication`

筛选标准：

- 优先纳入：后训练阶段消除/缓解后门、只需后门模型和干净数据、无需知道真实触发器的方法。
- 次级纳入：需要触发器反演或额外假设，但对 JSCC 防御设计有启发的方法。
- 排除：仅检测、不修复模型；训练期防御；需要毒化训练集或干净模型的方案。

本轮纳入核心文献 12 篇，其中通用 DNN 后门修复 9 篇，语义通信/JSCC 后门相关 3 篇。

## 3. 方法谱系

### 3.1 直接微调与剪枝类

代表工作：Fine-Pruning、BASS pruning defense。

核心思想是后门相关神经元在干净数据上激活较低或更脆弱，因此可以先剪枝，再用干净数据微调恢复主任务。Fine-Pruning 是最早的后门修复路线之一，指出单独剪枝或单独微调都不稳定，组合后可显著降低 ASR。

优点：

- 与当前假设高度匹配：只需要后门模型和干净数据。
- 实现简单，适合做 baseline。
- 对 BDJSCC 这类 CNN encoder-decoder 模型更容易落地。

局限：

- 后门神经元可能同时承担正常信道鲁棒性，过度剪枝会损害 clean PSNR。
- CT-BA 的后门触发位于 decoder 输入/信道统计域，未必表现为“干净输入低激活神经元”。
- ViT-JSCC 中“剪哪个 neuron/head/channel”需要重新定义。

对 CT-BA 的启发：

- 不能只照搬 encoder pruning。对于 CT-BA，后门映射可能主要体现在 decoder 对特定 latent/noise 分布的响应上，因此更合理的是 decoder-side pruning 或 decoder sensitivity pruning。
- 可以用 noise titration 暴露出的异常响应来定位敏感层/通道，而不是只用干净数据激活均值排序。

### 3.2 注意力蒸馏与表示对齐类

代表工作：Neural Attention Distillation (NAD)、Model-Contrastive Learning Defense (MCLDef)。

NAD 用干净数据先微调出 teacher，再让 backdoored student 对齐 teacher 的中间层 attention。MCLDef 则通过触发器反演构造“伪毒化样本”，再用模型对比学习把毒化特征拉回对应干净特征。

优点：

- 与“修复表示空间而不是只剪参数”这一目标更贴近。
- 对 ViT-JSCC 有潜力，因为 ViT 本身有 attention map，可直接构造 attention/feature distillation。
- 比单纯剪枝更适合保持主任务质量。

局限：

- NAD 仍依赖一个由同一干净数据微调得到的 teacher；若干净数据太少，teacher 的泛化有限。
- MCLDef 通常需要触发器反演，对 CT-BA 的信道触发器不一定可用。输入域反演不适合 channel-domain trigger。

对 CT-BA 的启发：

- 可以把 “teacher” 设为经过 clean-channel fine-tuning 的模型快照。
- 对齐对象不应只看分类 attention，而应看 JSCC 的 latent-to-image 重建路径：decoder feature maps、ViT decoder tokens、输出图像特征、oracle classifier feature。
- 可以用 noise titration 生成的随机 latent 作为“疑似触发域样本”，把其输出从高置信目标语义拉回低结构/低置信或 clean-like 分布。

### 3.3 对抗扰动与对抗遗忘类

代表工作：ANP、I-BAU、SAU。

ANP 发现后门模型在神经元对抗扰动下更容易坍缩到目标类，于是剪掉敏感神经元。I-BAU 将后门移除写成 clean data 上的 minimax 问题，用隐式超梯度求解。SAU 从后门风险和对抗风险的关系出发，生成共享对抗样本并对其遗忘。

优点：

- 与未知触发器设定匹配，不强依赖真实触发器。
- 思路上适合 CT-BA：CT-BA 本质就是 decoder 对某些 latent/channel perturbation 形成异常吸引子。
- 可以自然结合现有 noise titration：检测阶段已经找到了会激活后门的噪声强度区域。

局限：

- 原方法多数面向分类 loss，需要改成重建任务和语义置信目标。
- 对抗过程可能破坏 JSCC 的正常信道鲁棒性，需要严格约束 clean PSNR/SSIM。

对 CT-BA 的启发：

- 将“输入扰动”换成“decoder latent/channel perturbation”。
- 将分类 CE loss 换成混合目标：clean reconstruction loss + oracle confidence suppression + latent sensitivity regularization。
- 将 ANP 的 neuron perturbation 用于定位 decoder 中对 titration noise 高敏感的通道/token/head。

### 3.4 Unlearning-Recovering 与暴露后修复类

代表工作：RNP、EBYD。

RNP 先在少量干净数据上最大化错误以暴露后门神经元，再恢复正常性能并剪掉暴露出的后门神经元。EBYD 更进一步，把“暴露后门”作为统一预处理步骤，先通过 clean unlearning、稀疏化、权重扰动等方式放大隐藏后门功能，再接检测或修复。

优点：

- 非常适合当前任务，因为已经有 noise titration 检测，下一步可以把“检测信号”变成“暴露信号”。
- 不需要真实触发器，主要依赖后门模型与干净数据。
- 提供了一个合理的研究叙事：CT-BA 后门隐藏在正常信道鲁棒性中，必须先通过 channel/latent exposure 显性化，再修复。

局限：

- 暴露阶段如果做得太强，会损害主任务，导致后续恢复困难。
- 原论文仍主要在分类任务评估，迁移到重建任务需要新的指标。

对 CT-BA 的启发：

- 可以提出 Channel-Aware Backdoor Exposure：用 noise titration 发现异常 r 区间，再对模型做轻量 clean unlearning 或 latent perturbation exposure。
- 后续修复可以选择剪枝、微调、蒸馏或 sensitivity regularization。
- 这是最有论文潜力的方向，因为它将现有 detection 工作自然推进到 removal。

### 3.5 Sharpness-aware 微调类

代表工作：FT-SAM。

FT-SAM 认为普通 fine-tuning 难以让后门相关神经元逃离局部极小值，而 SAM 可以通过平坦化局部损失区域来压缩后门相关神经元范数，从而增强干净数据微调的修复能力。

优点：

- 与当前假设高度匹配，工程实现相对直接。
- 可作为比 vanilla fine-tuning 更强的 baseline。
- 对 ViT-JSCC 和 BDJSCC 都可尝试。

局限：

- SAM 计算量约为普通训练的两倍。
- 需要定义适合 JSCC 的 clean loss 和 backdoor suppression loss。
- 如果只对 clean reconstruction 做 SAM，可能修复不彻底；应结合 titration exposure。

对 CT-BA 的启发：

- baseline：clean data 上 vanilla fine-tuning vs FT-SAM。
- 改进：Channel-aware FT-SAM，在 clean reconstruction loss 外加入 decoder latent noise consistency。

## 4. 代表文献表

| 工作 | 类型 | 所需资源 | 是否匹配当前假设 | 对 CT-BA/JSCC 的价值 |
|---|---|---:|---|---|
| Fine-Pruning (Liu et al., 2018) | 剪枝 + 微调 | 后门模型 + 干净数据 | 高 | 基础 baseline；可做 decoder/channel pruning 变体 |
| Neural Cleanse (Wang et al., 2019) | 触发器反演 + 修复 | 后门模型 + 干净数据 + 离散标签 | 中低 | 输入域触发器假设不适合 CT-BA，但可作为反演类对照 |
| NAD (Li et al., 2021) | attention distillation | 后门模型 + 少量干净数据 | 高 | 适合 ViT-JSCC 的 token/attention 对齐 |
| ANP (Wu & Wang, 2021) | 对抗神经元剪枝 | 后门模型 + 极少干净数据 | 高 | 可迁移为 decoder adversarial channel pruning |
| I-BAU (Zeng et al., 2022) | minimax adversarial unlearning | 后门模型 + 少量干净数据 | 高 | 可改成 latent/channel-domain unlearning |
| MCLDef (Yue et al., 2022) | trigger inversion + contrastive repair | 后门模型 + 干净数据 + 反演触发器 | 中 | 表示对齐思路有价值，但输入触发反演需替换 |
| RNP (Li et al., 2023) | unlearn-recover-prune | 后门模型 + 少量干净数据 | 高 | 很适合设计 CT-BA exposure-removal pipeline |
| SAU (Wei et al., 2023) | shared adversarial unlearning | 后门模型 + 小干净集 | 高 | 可把 shared adversarial examples 换成 shared latent perturbations |
| FT-SAM (Zhu et al., 2023) | SAM fine-tuning | 后门模型 + benign data | 高 | 强 baseline；可与 titration regularization 结合 |
| EBYD (Li et al., 2024) | expose before defend | 后门模型 + 干净数据 | 高 | 适合作为研究框架，支撑“先暴露再消除”的叙事 |
| BASS defense (Zhou et al., 2024) | 语义通信 pruning/reverse engineering | 训练数据/模型/输入域触发 | 中 | JSCC 场景直接相关，但威胁模型与 CT-BA 不同 |
| CT-BA (Wan et al., 2025) | channel-triggered attack + titration detection | 后门模型 + oracle | 背景工作 | 明确指出 detection 不等于 removal，是本文研究空白 |

## 5. 与当前 CT-BA 防御问题的差距

现有通用后门消除方法大多默认：

- 任务是分类。
- 触发器在输入图像域。
- 后门行为表现为目标类别预测。
- 修复指标是 clean accuracy 和 ASR。

CT-BA/JSCC 的关键差异是：

- 任务是连续图像重建。
- 触发器在 channel/latent 域，而不是输入图像域。
- 后门行为是 decoder 输出目标图像或目标语义。
- 需要同时约束 PSNR/SSIM、oracle confidence、noise titration 曲线、latent robustness。

因此，直接套用 Neural Cleanse、MCLDef 等输入域触发器反演方法不合适。更有希望的是 clean-data model repair 路线：Fine-Pruning、NAD、ANP、I-BAU、RNP、FT-SAM、EBYD。它们共同点是无需真实触发器，能在后训练阶段利用干净数据修复模型。

## 6. 初步研究机会

### 6.1 Channel-Aware Backdoor Exposure and Removal

研究问题：

> 给定已检测为后门的 JSCC 模型和干净数据，能否通过 channel/latent-domain exposure 显性化后门敏感区域，并在不显著降低 clean reconstruction quality 的前提下消除 CT-BA？

核心流程：

1. 使用 noise titration 找到异常噪声强度区间 `R_bad`。
2. 对 decoder 输入注入 `R_bad` 中的 latent noise，记录异常输出和高置信 oracle response。
3. 执行轻量 exposure：clean unlearning、weight perturbation 或 latent adversarial perturbation。
4. 定位敏感参数：decoder channel/token/head/filter 的 sensitivity score。
5. 修复模型：剪枝 + clean reconstruction fine-tuning，或 distillation + sensitivity regularization。

### 6.2 Titration-Regularized Fine-Tuning

研究问题：

> 能否在 clean data fine-tuning 中加入 noise titration regularization，使模型对随机 latent/channel noise 不再产生高置信目标语义？

可能 loss：

```text
L = L_clean_recon
  + lambda_1 * L_noise_conf_suppression
  + lambda_2 * L_decoder_consistency
  + lambda_3 * L_weight_drift
```

其中：

- `L_clean_recon`：干净图像通过正常信道后的 MSE/LPIPS/SSIM loss。
- `L_noise_conf_suppression`：随机 latent noise 解码图像经过 oracle 后，不应产生高置信稳定类别。
- `L_decoder_consistency`：正常 latent 的小扰动不应剧烈改变语义。
- `L_weight_drift`：限制模型远离原始后门模型太多，以保留主任务。

### 6.3 Decoder-Side ANP/RNP

研究问题：

> CT-BA 后门是否集中在 decoder 的少量敏感通道、token 或 attention heads 中？

做法：

- 对 decoder 参数施加 adversarial perturbation。
- 用 clean data 和 titration noise 同时评估 collapse/sensitivity。
- 剪掉或冻结高 sensitivity 单元。
- 再进行 clean fine-tuning。

这条路线最容易产生直观实验图：消除前后 titration curve、target reconstruction、clean PSNR 对比。

## 7. 推荐实验路线

建议先做三层 baseline，再做一个主方法。

Baseline 1：Vanilla clean fine-tuning

- 只用干净数据继续训练后门模型。
- 验证普通微调能否降低 titration ASR/π 曲线。
- 这是必要下限。

Baseline 2：Fine-Pruning / Decoder-Pruning + fine-tuning

- 对 BDJSCC 先做 decoder convolution channel pruning。
- 对 ViT-JSCC 后续再考虑 token/head pruning。
- 指标：clean PSNR drop、titration confidence drop。

Baseline 3：FT-SAM

- 在干净重建 loss 上做 SAM fine-tuning。
- 验证 sharpness-aware fine-tuning 是否比普通 fine-tuning 更能压制后门。

Main method：Titration-Regularized Backdoor Unlearning

- 使用 noise titration 生成 backdoor exposure samples。
- 对这些 samples 优化“不要输出稳定高置信语义/目标图像”的损失。
- 同时用干净数据保持主任务。
- 如果需要进一步增强，可叠加 decoder sensitivity pruning。

## 8. 评估指标建议

后门消除效果：

- `π_r^χ` 曲线下降：与检测阶段一致。
- Titration ASR：随机 latent 解码图像被 oracle 判为目标类的比例。
- Target PSNR/SSIM：如果有目标图像，消除后应显著下降。
- Triggered semantic confidence：触发域输出的最大 softmax 置信度应下降或分散。

主任务保持：

- clean PSNR / SSIM / LPIPS。
- oracle CA 或 AEVC。
- 正常 SNR/channel 下的 reconstruction quality。

副作用：

- 过度修复导致输出模糊。
- 正常低 SNR 鲁棒性下降。
- 对不同 r、不同 chi、不同随机种子的稳定性。

## 9. 当前最适合写成论文贡献的方向

最推荐的研究表述：

> We propose a channel-aware post-training backdoor removal framework for JSCC semantic image reconstruction. Unlike existing backdoor repair methods designed for input-domain classification triggers, our method uses noise titration to expose channel-triggered backdoor behavior, then removes the abnormal decoder response through clean-data constrained unlearning and sensitivity regularization.

潜在贡献点：

1. 第一个面向 channel-triggered JSCC reconstruction backdoor 的 post-training removal 框架。
2. 将 noise titration 从 detection 扩展为 exposure-guided unlearning。
3. 提出适合连续重建任务的后门消除指标，而不是只用分类 ASR/ACC。
4. 在 BDJSCC 和 ViT-JSCC 上验证 clean utility 与 backdoor removal 的 trade-off。

## 10. 参考文献

- Gao, Y., Xu, C., Wang, D., Chen, S., Ranasinghe, D. C., & Nepal, S. (2019). STRIP: A Defence Against Trojan Attacks on Deep Neural Networks. arXiv:1902.06531. https://arxiv.org/abs/1902.06531
- Li, Y., Huang, H., Zhang, J., Ma, X., & Jiang, Y.-G. (2024). Expose Before You Defend: Unifying and Enhancing Backdoor Defenses via Exposed Models. arXiv:2410.19427. https://arxiv.org/abs/2410.19427
- Li, Y., Lyu, X., Koren, N., Lyu, L., Li, B., & Ma, X. (2021). Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks. arXiv:2101.05930. https://arxiv.org/abs/2101.05930
- Li, Y., Lyu, X., Ma, X., Koren, N., Lyu, L., Li, B., & Jiang, Y.-G. (2023). Reconstructive Neuron Pruning for Backdoor Defense. arXiv:2305.14876. https://arxiv.org/abs/2305.14876
- Liu, K., Dolan-Gavitt, B., & Garg, S. (2018). Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. arXiv:1805.12185. https://arxiv.org/abs/1805.12185
- Wan, J., Cheng, N., & Shen, J. (2025). A Channel-Triggered Backdoor Attack on Wireless Semantic Image Reconstruction. arXiv:2503.23866. https://arxiv.org/abs/2503.23866
- Wang, J., Hassan, G. M., & Akhtar, N. (2022). A Survey of Neural Trojan Attacks and Defenses in Deep Learning. arXiv:2202.07183. https://arxiv.org/abs/2202.07183
- Wei, S., Zhang, M., Zha, H., & Wu, B. (2023). Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples. arXiv:2307.10562. https://arxiv.org/abs/2307.10562
- Wu, D., & Wang, Y. (2021). Adversarial Neuron Pruning Purifies Backdoored Deep Models. arXiv:2110.14430. https://arxiv.org/abs/2110.14430
- Yue, Z., Xia, J., Ling, Z., Hu, M., Wang, T., Wei, X., & Chen, M. (2022). Model-Contrastive Learning for Backdoor Defense. arXiv:2205.04411. https://arxiv.org/abs/2205.04411
- Zeng, Y., Chen, S., Park, W., Mao, Z. M., Jin, M., & Jia, R. (2022). Adversarial Unlearning of Backdoors via Implicit Hypergradient. arXiv:2110.03735. https://arxiv.org/abs/2110.03735
- Zhou, Y., Hu, R. Q., & Qian, Y. (2024). Backdoor Attacks and Defenses on Semantic-Symbol Reconstruction in Semantic Communications. arXiv:2404.13279. https://arxiv.org/abs/2404.13279
- Zhu, M., Wei, S., Shen, L., Fan, Y., & Wu, B. (2023). Enhancing Fine-Tuning Based Backdoor Defense with Sharpness-Aware Minimization. arXiv:2304.11823. https://arxiv.org/abs/2304.11823
