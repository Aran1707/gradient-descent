# Student Interaction Questions

## Warm-up

1. For `f(x)=x^2`, what does `f'(x)` tell us at `x=3`?
2. If `f(x,y)=x^2+y^2`, what are the two partial derivatives?
3. At `(1,2)`, which direction does the gradient point?

## Prediction exercise

Given:

```text
theta = (1, 2)
gradient = (2, 4)
learning_rate = 0.1
```

Ask students to compute the next point before showing it.

Answer:

```text
(0.8, 1.6)
```

## Concept checks

1. Does a zero gradient always mean we found a minimum?
2. Why can a larger learning rate make training worse?
3. Why is `x^2+25y^2` harder for gradient descent than `x^2+y^2`?
4. What does momentum remember?
5. What do Adam's first and second moments roughly represent?
6. Why is AdamW not just a cosmetic rename of Adam?

## Discussion prompts

1. If an optimizer trains faster but generalizes worse, is it better?
2. Why should we compare optimizers across multiple random seeds?
3. Why might a 2D toy result not transfer directly to a neural network?
