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

Which means $\hat{C}(r) = \sum\limits_{i=1}^NT_i(1-exp(-\sigma_i\delta_i))c_i$, where $T_i=exp\left(-\sum\limits_{j=1}^{i-1}\sigma_j\delta_j\right)$ in nerf paper. (accumulated transmittance in graphics).

### 3/14 Tue.

As mentioned above, $T_i=exp\left(-\sum\limits_{j=1}^{i-1}\sigma_j\delta_j\right)$ is the accumulated transmittance, however $T_1$ is undefined because $i-1=0<1$ when $i = 1$, consider it as accumulated term, we can set the first element in the tensor to 1 before no $\sigma_j$ and $\delta_j$ is accumulated before it.

<img src="./images/volumetric result.png" alt="output" style="zoom:50%;" />

Got a better quality!

> volumetric rendering finished now!

Now I am trying to optimize the *color* so that it 'converges' to the **RED** color of my sphere model.

To use gradient descent, 

1. Set the tensor to optimize using `requires_grad=True` 
2. Call `tensor.backward()`, if we do not call this backward() method then gradients are not calculated for the tensors having ***required_grad*** set to ***True***.

#### How are `optimizer.step()` and `loss.backward()`related?

- `optimizer.step` is performs a parameter update based on the *current* gradient (stored in `.grad` attribute of a parameter) and the update rule.
- Calling `.backward()` mutiple times accumulates the gradient (by addition) for each parameter. This is why you should call `optimizer.zero_grad()` after each `.step()` call.

<img src="./images/volumetric rendering optimize/first_10.png" alt="first_10" style="zoom: 33%;" /><img src="./images/volumetric rendering optimize/second_10.png" alt="second_10" style="zoom: 33%;" /><img src="./images/volumetric rendering optimize/third_10.png" alt="third_10" style="zoom:33%;" />

As shown above, I got a great result from my toy optimizer ;)

#### Transformation matrix

In previous volumetric rendering, I set the focal point as origin, however, it isn't actually the origin in world coordinates. So a transformation matrix is necessary to convert the camera coordinates to world coordinates.

It looks like:
$$
M=
\begin{pmatrix}
a_{11} & a_{12} & a_{13} & a_{14}\\
a_{21} & a_{22} & a_{23} & a_{24}\\
a_{31} & a_{32} & a_{33} & a_{34}\\
0 & 0 & 0 & 1
\end{pmatrix}
$$
The top-right 3*1 matrix is the real coordinate of focal point.
