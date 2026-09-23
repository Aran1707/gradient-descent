# Math Notes

## Gradient of `f(x,y)=x^2+y^2`

\[
f(x,y)=x^2+y^2
\]

\[
\frac{\partial f}{\partial x}=2x,\qquad
\frac{\partial f}{\partial y}=2y
\]

\[
\nabla f(x,y)=\begin{bmatrix}2x\\2y\end{bmatrix}
\]

At `(1,2)`:

\[
\nabla f(1,2)=\begin{bmatrix}2\\4\end{bmatrix}
\]

With learning rate `eta=0.1`:

\[
\theta_1=\begin{bmatrix}1\\2\end{bmatrix}
-0.1\begin{bmatrix}2\\4\end{bmatrix}
=\begin{bmatrix}0.8\\1.6\end{bmatrix}
\]

Function value drops from `5` to `0.8^2+1.6^2=3.2`.

## Directional derivative and steepest direction

For a unit vector `u`,

\[
D_u f=\nabla f\cdot u.
\]

By Cauchy–Schwarz,

\[
|\nabla f\cdot u|\leq \|\nabla f\|\|u\|=\|\nabla f\|.
\]

So the maximum directional derivative occurs when `u` points in the gradient direction. The most negative directional derivative occurs when `u` points opposite the gradient.

That is the mathematical reason for gradient descent.

## Taylor argument for descent

For small step `delta`,

\[
f(\theta+\delta)\approx f(\theta)+\nabla f(\theta)^T\delta.
\]

Choose

\[
\delta=-\eta\nabla f(\theta).
\]

Then

\[
f(\theta-\eta\nabla f(\theta))\approx f(\theta)-\eta\|\nabla f(\theta)\|^2.
\]

If `eta > 0` is small enough and the gradient is nonzero, the function should locally decrease.

## Ill-conditioned quadratic

\[
f(x,y)=x^2+25y^2
\]

\[
\nabla f=\begin{bmatrix}2x\\50y\end{bmatrix},\qquad
H=\begin{bmatrix}2&0\\0&50\end{bmatrix}
\]

Gradient descent:

\[
x_{t+1}=(1-2\eta)x_t,
\qquad
 y_{t+1}=(1-50\eta)y_t.
\]

Convergence requires both magnitudes to be smaller than one:

\[
|1-2\eta|<1,
\qquad
|1-50\eta|<1.
\]

So:

\[
0<\eta<1,
\qquad
0<\eta<0.04.
\]

The stricter condition is:

\[
0<\eta<0.04.
\]

This is why the steep `y` direction limits the learning rate.

## Local minimum example

Use:

\[
f(x,y)=(x^2-1)^2+0.2x+y^2.
\]

This is nonconvex and has different basins of attraction depending on the starting point. Use this for local/global minimum discussion, not `x^2+25y^2`.

## Saddle example

\[
f(x,y)=x^2-y^2
\]

\[
\nabla f=\begin{bmatrix}2x\\-2y\end{bmatrix}
\]

At `(0,0)`, the gradient is zero, but the point is not a minimum. It is a saddle.
