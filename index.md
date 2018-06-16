---
layout: default
title: {{ site.name }}
---

---

# Abstract

With the success of deep learning based approaches in tackling challenging problems in computer vision, a wide range of deep architectures have recently been proposed for the task of visual odometry (VO) estimation. Most of these proposed solutions rely on supervision, which requires the acquisition of precise ground-truth camera pose information, collected using expensive motion capture systems or high-precision IMU/GPS sensor rigs. In this work, we propose an unsupervised paradigm for deep visual odometry learning. We show that using a noisy teacher, which could be a standard VO pipeline, and by designing a loss term that enforces geometric consistency of the trajectory, we can train accurate deep models for VO that do not require ground-truth labels. We leverage geometry as a self-supervisory signal and propose "Composite Transformation Constraints (CTCs)", that automatically generate supervisory signals for training and enforce geometric consistency in the VO estimate. We also present a method of characterizing the uncertainty in VO estimates thus obtained. To evaluate our VO pipeline, we present exhaustive ablation studies that demonstrate the efficacy of end-to-end, self-supervised methodologies to train deep models for monocular VO. We show that leveraging concepts from geometry and incorporating them into the training of a recurrent neural network results in performance competitive to supervised deep VO methods.

---

# Intuition

![alt text]( {{ site.baseurl }}/assets/CTCNet_intuition.png "Intuition of Composite Transformation Constraints" ){:width="40%"}

We leverage the observation that compounded sequences of transformations over short timescales should be equivalent to a single transformation independently computed over longer timescales. This allows us to create Composite Transformation Constraints (CTCs) that can be used as supervisory signals for learning visual odometry.

---

# CTCNet

![alt text]( {{ site.baseurl }}/assets/CTCNet_architecture.png "CTCNet architecture" )

End-to-end architecture: An example of Composite Transformation Constraints (CTCs) being applied to 4 successive input images. During training, two estimates are generated from the inputs: one for a sequential pairwise constraint and one for a CTC constraint. At test time, each frame is only fed into the network once to receive the output pose from the SE(3) layer.

---

