---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Paper Title
---

# ABSTACT
Robotic soft tissue manipulation in surgery presents significant challenges due to the tissue's high deformability and the spatial constraints of the surgical environment. To address these challenges, we propose a novel framework that combines a deformation model-based shape controller with an analytical approach for contact point selection, employing manipulability-based affordance evaluation. Unlike data-driven affordance methods, our approach leverages the deformation Jacobian matrix derived from a linearized deformation model to evaluate the manipulability of candidate contact points under specified contact conditions, providing a robust and analytical solution for affordance estimation in soft tissue manipulation. For deformation control, we employ a differentiable deformation model based on Projective Dynamics (PD) to efficiently compute forward and backward deformation processes in real-time. Point-based visual features are constructed to represent and track the desired tissue deformation, enabling precise manipulation through visual feedback. The proposed framework has been validated through simulations and physical experiments. These results demonstrate the desired deformation effects and a strong correlation between the predicted affordance and the observed manipulation efficiency, confirming the effectiveness of our approach in soft tissue deformation control and affordance estimation.


# METHOD
<!-- ![Control framework](images/control-frame.png) -->
<img src="images/control-frame.png" style="display: block; max-width: 95%; height: auto; margin-left: auto; margin-right: auto;" alt="Control framework">

# SIMULATIONS
### 1. Control of a Specified Angle

<div style="width: 95%; margin: 20px auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/ZGHS-rUw-EA?si=7K0D-2aBA6oxVnK8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

<div style="width: 95%; margin: 20px auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/FYNRn9j0Kcw?si=exoqHK5uEJf8byYV" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

<div style="width: 95%; margin: 20px auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/B0zZOhVUEC0?si=KDfZzktnTZMkDfUd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

### 2. Liver Retraction for Gallbladder Exposure

<div style="width: 95%; margin: 20px auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/sLFUk8dWhxg?si=V-ZRhEy0FQocaVQp" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

# EXPERIMENTS

## Pre-tension
### 1. Silicon Material

<div style="width: 95%; margin: 20px auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/W5ah8saajTQ?si=IpVgW3Zm8S_paimM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

<div style="width: 95%; margin: 0 auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/QkQ69AMHQvE?si=RsiFIO-3M3N05MI3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>


### 2. Latex Material
<div style="width: 95%; margin: 20px auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/F__YF0XvHFc?si=GWB0R2Kydyery7pC" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

<div style="width: 95%; margin: 20px auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/Zhe9zo_uGPw?si=NPzpI23AkaxMDyPY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Tissue Retration
<div style="width: 95%; margin: 20px auto;">
   <iframe width="640" height="360" src="https://www.youtube.com/embed/xpktfKmVkwg?si=xlmK6UuVCRZtYAaB" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>
