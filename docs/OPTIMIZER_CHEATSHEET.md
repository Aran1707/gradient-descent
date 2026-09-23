# Optimizer Cheatsheet

## Gradient Descent

\[
\theta_t=\theta_{t-1}-\eta \nabla L(\theta_{t-1})
\]

Use as the base case.

## Mini-batch SGD

\[
g_t=\frac{1}{|B_t|}\sum_{i\in B_t}\nabla_\theta \ell_i(\theta_t)
\]

\[
\theta_{t+1}=\theta_t-\eta g_t
\]

The gradient is estimated from a mini-batch rather than the entire dataset.

## Momentum

\[
v_t=\beta v_{t-1}+g_t
\]

\[
\theta_t=\theta_{t-1}-\eta v_t
\]

Intuition: remember consistent directions and smooth noisy/oscillating gradients.

## Nesterov Momentum

\[
v_t=\beta v_{t-1}+\nabla L(\theta_{t-1}-\eta\beta v_{t-1})
\]

\[
\theta_t=\theta_{t-1}-\eta v_t
\]

Intuition: look ahead before computing the correction.

## AdaGrad

\[
G_t=G_{t-1}+g_t\odot g_t
\]

\[
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{G_t}+\epsilon}
\]

Good teaching point: effective learning rates shrink over time.

## RMSProp

\[
v_t=\rho v_{t-1}+(1-\rho)g_t^2
\]

\[
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}
\]

Uses an exponential moving average instead of AdaGrad's cumulative sum.

## Adam

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]

\[
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

Teach the bias correction: both moving averages start at zero, so early values are biased downward.

## AdamW

Core idea: decouple weight decay from the adaptive gradient update.

One simplified form:

\[
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

Do not simply say AdamW is “Adam but better.” Say what changed mathematically.

## Lion

Ignoring weight decay:

\[
c_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
\theta_t=\theta_{t-1}-\eta\operatorname{sign}(c_t)
\]

\[
m_t=\beta_2m_{t-1}+(1-\beta_2)g_t
\]

Teaching point: update magnitude is controlled by the sign operation rather than raw gradient magnitude.

## SAM

\[
\min_\theta \max_{\|\epsilon\|\leq \rho} L(\theta+\epsilon)
\]

Teaching point: search for a parameter region with low loss nearby, not only a point with low loss.

Expensive for your NumPy CNN because it needs approximately two gradient evaluations per step.
