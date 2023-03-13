## NeRF
### 3/3 Fri.
Try to implement ray tracing method in camera_model.
### 3/6 Mon.
Finish doing ray tracing by plotting a sphere.
### 3/7 Tue.
Though realistic 3D models are complex, we can break down a model into infinite number of triangles and do intersect to render a 2D image(triangle mesh). But it is not differentiable and hard to optimize.

So NeRF uses volumetric rendering which is easier to optimize. $i.e.,\ (r,g,b,\sigma)$

### 3/8 - 3/12

Doing CS 110 Project 1.1...

### 3/13 Mon.

In volumetric rendering, `rendering()`  function, I implemented the discretized integration to compute the color and density in rays.

```python
alpha = 1 - torch.exp(-density * delta.unsqueeze(0)) # [num_rays, num_bins, 1]
T = compute_accumulated_transmittance(1 - alpha)
```

Which means $\hat{C}(r) = \sum\limits_{i=1}^NT_i(1-exp(-\sigma_i\delta_i))c_i$, where $T_i=exp\left(-\sum\limits_{j=1}^{i=1}\sigma_j\delta_j\right)$ in nerf paper. (accumulated transmittance in graphics).
