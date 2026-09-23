# Talk Script Notes

## Opening

“Today we are not trying to memorize ten optimizer names. We are trying to understand one idea: if a function tells us which direction goes uphill fastest, can we use that information to go downhill?”

## Gradient section

Emphasize that a gradient is not just “the derivative in 2D.” It is a vector that assigns a direction and magnitude to each point.

Use the hand calculation before showing the figures. The visual should confirm the math, not replace it.

## Learning-rate section

Say explicitly:

“`x^2+25y^2` is not hard because it has many minima. It is hard because the surface curves much more strongly in one direction than the other.”

## Optimizer section

Introduce every optimizer as an answer to a problem:

- SGD: full gradients are expensive.
- Momentum: SGD is noisy and can zigzag.
- AdaGrad/RMSProp: different parameters may need different effective step sizes.
- Adam: combine momentum-like direction and adaptive scaling.
- AdamW: make weight decay behave correctly with adaptive updates.
- Lion: what if we use sign-based momentum updates?
- SAM: what if we care about flat neighborhoods, not only point loss?

## NumPy CNN section

Recommended line:

“This final demo is deliberately not PyTorch. The CNN was implemented with NumPy, so the same mathematical pieces we discussed — forward pass, gradients, and parameter updates — are visible in code.”

Then do not drown them in convolution internals. Show architecture, training curves and one optimizer update snippet.
