---
layout: page
title: "Mode-Guided Dataset Distillation using Diffusion Models"
description: "ICML 2025 Oral (Top 1.0%) — A novel approach for dataset distillation that boosts diversity and performance without fine-tuning."
image_small: /assets/img/mgd3/banner.png
image_large: /assets/img/mgd3/banner.png
image: assets/img/mgd3/banner.png
github: https://github.com/jachansantiago/mode_guidance
redirect: https://jachansantiago.com/mode-guided-distillation/
importance: 1
horizontal: true
---

<div class="links" style="margin-bottom: 1.5rem;">
  <a href="https://arxiv.org/pdf/2505.18963.pdf" target="_blank">📄 Paper</a> &nbsp;|&nbsp;
  <a href="https://github.com/jachansantiago/mode_guidance" target="_blank">💻 Code</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2505.18963" target="_blank">📝 arXiv</a>
</div>

**Authors:**
[Jeffrey A. Chan Santiago](https://jachansantiago.com/)<sup>1</sup>,
[Praveen Tirupattur](https://ptirupat.github.io/)<sup>1</sup>,
[Gaurav Kumar Nayak](https://sites.google.com/view/gauravnayak/)<sup>2</sup>,
[Gaowen Liu](https://scholar.google.com/citations?user=NIv_aeQAAAAJ&hl=en)<sup>3</sup>,
[Mubarak Shah](https://www.crcv.ucf.edu/person/mubarak-shah/)<sup>1</sup>

<sup>1</sup>Center for Research in Computer Vision, University of Central Florida &nbsp;
<sup>2</sup>Mehta Family School of DS & AI, Indian Institute of Technology Roorkee, India &nbsp;
<sup>3</sup>Cisco Research

---

## Abstract

**Dataset distillation** has emerged as an effective strategy, significantly reducing training costs and facilitating more efficient model deployment. Recent advances have leveraged generative models to distill datasets by capturing the underlying data distribution.

Unfortunately, existing methods require model fine-tuning with distillation losses to encourage diversity and representativeness. However, these methods do not guarantee sample diversity, limiting their performance.

We propose a *mode-guided diffusion model* that leverages a pre-trained diffusion model without the need for fine-tuning using distillation losses. Our approach addresses dataset diversity in three stages: **Mode Discovery** to identify distinct data modes, **Mode Guidance** to enhance intra-class diversity, and **Stop Guidance** to mitigate artifacts in synthetic samples that affect performance.

We evaluate our approach on *ImageNette*, *ImageIDC*, *ImageNet-100*, and *ImageNet-1K*, achieving accuracy improvements of 4.4%, 2.9%, 1.6%, and 1.6%, respectively, over state-of-the-art methods. Our method eliminates the need for fine-tuning diffusion models with distillation losses, significantly reducing computational costs.

---

## The Task

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/mgd3/task.png' | relative_url }}" alt="Overview of dataset distillation task" title="Dataset Distillation Task"/>
    </div>
</div>
<div class="caption">
    Dataset distillation aims to compress the knowledge of a large training dataset into a significantly smaller set of synthetic samples, such that models trained on this distilled dataset can achieve performance comparable to those trained on the full dataset.
</div>

- **Optimization-based Distillation:** Learns a synthetic dataset by directly optimizing it to match the gradient or feature statistics of the original dataset.
- **Generation-based Distillation:** First models the distribution of the original dataset, then generates samples that approximate this learned distribution.

---

## Motivation

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/mgd3/gradient.png' | relative_url }}" alt="Overview of gradient field in diffusion" title="Gradient Field Overview"/>
    </div>
</div>
<div class="caption">
    Overview of the gradient field (score function) during the denoising process in latent diffusion for a specific class <em>c</em>.
</div>

The original data distribution (blue dots) highlights denser regions via an orange gradient field. To generate a sample, noise is initially sampled from a standard normal distribution.

- **(a) DiT:** A pre-trained diffusion model without fine-tuning leads to imbalanced mode likelihoods, resulting in limited sample diversity and frequent repetition of modes.
- **(b) MinMax Diffusion:** Fine-tunes the model to balance mode likelihoods and improve diversity. However, it still suffers from sample redundancies tied to initial noise conditions.
- **(c) MGD³ (Ours):** Introduces mode-guided denoising (colored traces), explicitly steering samples toward distinct modes (stars). After *k* guided steps, it transitions to unguided denoising (black trace), achieving both high diversity and consistency—without requiring any fine-tuning.

---

## Our Method

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/mgd3/test.png' | relative_url }}" alt="Overview of the proposed method" title="MGD³ Method Overview"/>
    </div>
</div>
<div class="caption">
    Overview of the proposed method for distilled dataset synthesis using a diffusion model. The approach consists of three stages: <em>Mode Discovery</em>, <em>Mode Guidance</em>, and <em>Stop Guidance</em>.
</div>

- **Mode Discovery:** Estimates the *N* modes of the original dataset in the latent diffusion model's generative space.
- **Mode Guidance:** Given a mode *m_k* and class *c*, the generation process is steered toward the mode *m_k* for *t_stop* denoising steps using the pre-trained model.
- **Stop Guidance:** After *t_stop* steps, the model transitions to standard unguided denoising. Without guidance, generations may follow the unguided path, resulting in redundant or overlapping samples.

---

## Results

### ImageNet Subsets

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/mgd3/table.png' | relative_url }}" alt="Table 1: ImageNet subset results" title="ImageNet Subset Results"/>
    </div>
</div>
<div class="caption">
    Comparison of performance between pre-trained diffusion models and state-of-the-art methods on ImageNet subsets. Evaluated using the hard-label protocol with ResNet-10 and average pooling.
</div>

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/mgd3/table2.png' | relative_url }}" alt="Table 2: Cross-architecture comparison" title="Cross-Architecture Comparison"/>
    </div>
</div>
<div class="caption">
    Comparison with generative prior methods. Evaluation across architectures (AlexNet, VGG11, ResNet18, ViT) and ImageNet subsets (A–E) using the hard-label protocol. Our method outperforms GLaD, H-GLaD, and LM3D in the cross-architecture setup.
</div>

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/mgd3/imagenet1k.png' | relative_url }}" alt="ImageNet-1K results" title="ImageNet-1K Results"/>
    </div>
</div>
<div class="caption">
    Comparison with state-of-the-art methods on ImageNet-1K using the soft-label protocol. Our method achieves state-of-the-art performance, outperforming prior approaches by 1.3% and 1.6% on IPC 10 and IPC 50, respectively.
</div>

### Text-to-Image Diffusions

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/mgd3/text-to-image.png' | relative_url }}" alt="Text-to-Image results" title="Text-to-Image Results"/>
    </div>
</div>
<div class="caption">
    Performance of the Text-to-Image model across multiple datasets using the soft-label protocol. Mode guidance significantly improves performance over Stable Diffusion across all datasets, including gains of 3.4% and 2.3% on ImageNet-1K at IPC 10 and IPC 50, respectively.
</div>

### Ablations

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/mgd3/ablations.png' | relative_url }}" alt="Ablation study" title="Ablation Study"/>
    </div>
</div>
<div class="caption">
    Ablation study on the components of the proposed method. Evaluated on ImageNette with IPC 10. Each component contributes to performance gains, with Stop Guidance playing a key role in enhancing final accuracy.
</div>

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{chan2025mgd3,
  title     = {{MGD}$^3$: Mode-Guided Dataset Distillation using Diffusion Models},
  author    = {Chan Santiago, Jeffrey A. and Tirupattur, Praveen and Nayak, Gaurav Kumar and Liu, Gaowen and Shah, Mubarak},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
}
```
{% endraw %}