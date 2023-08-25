# Explanation of Dense Neural Network Layer

The Dense network implemented in `Dense.py` implements one key improvement over the Basic dense network in `Basic.py`, the **Adam optimization algorithm**.

### Introduction to Adam:

**Adam** (short for **Adaptive Moment Estimation**) is a popular optimization algorithm used in training machine learning models, especially for deep neural networks. It combines the advantages of both the **Adagrad** and **RMSProp** optimization algorithms, offering adaptive learning rates and momentum-based updates. The algorithm was introduced by Diederik P. Kingma and Jimmy Ba in their paper titled "Adam: A Method for Stochastic Optimization."

Adam maintains a running estimate of the first moment (mean) and the second moment (uncentered variance) of the gradients with respect to each parameter during training. These moments are used to adaptively adjust the learning rate for each parameter. This adaptive learning rate helps in speeding up convergence and handling the scaling of different parameters.

### Adam Algorithm:

Here's a breakdown of how the Adam algorithm works:

1. **Initialization**:
   Adam maintains two moving averages for each parameter: the first moment estimate (mean) and the second moment estimate (uncentered variance). These moving averages are initialized to zero at the start.

2. **Compute Gradients**:
   In each iteration of the optimization process, you compute the gradients of the model's parameters with respect to the loss function using techniques like backpropagation.

3. **Update Moving Averages**:
   The moving averages are updated using exponential moving averages of the gradient and its square. For a parameter $ \theta $ and its gradient $ g $, the moving averages are updated as follows:

    $$ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$
    $$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $$

    Here, $ \beta_1 $ and $ \beta_2 $ are hyperparameters that control the decay rates of the moving averages. Typically, $ \beta_1 $ is set to 0.9 and $ \beta_2 $ is set to 0.999.

4. **Bias Correction**:
   Since the moving averages are initialized to zero, they can be biased towards zero, especially in the early iterations. To counteract this bias, bias correction steps are applied to the moving averages:

    $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$
    $$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

    Here, $ t $ represents the iteration number.

5. **Update Parameters**:
   Finally, the model's parameters are updated using the bias-corrected moving averages. The learning rate $ \alpha $ is another hyperparameter controlling the step size of the updates.

    $$ \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t $$

    Here, $ \epsilon $ is a small constant (e.g., $ 10^{-8} $) added to the denominator to prevent division by zero.

The Adam algorithm combines the benefits of adapting the learning rates per parameter (AdaGrad) and utilizing the moving average of past squared gradients (RMSProp), along with the concept of momentum. This combination enables faster convergence and better handling of sparse gradients, making it a widely used optimization algorithm for training deep neural networks.

### Initialization of Adam Parameters:

```python
class Dense(Layer):
    def __init__(self, o_size: int, i_size: int = -1, activation: str = "none",
                 eta: float = 0.01, t0: float = 1, dt: float = 0.0001):
        super().__init__()

        self.t = t0  # Time step counter
        self.dt = dt  # Time step size

        self.B1 = 0.9  # Exponential decay rate for the first moment estimates
        self.B2 = 0.999  # Exponential decay rate for the second moment estimates
        self.eps = 10 ** -8  # Small constant for numerical stability

        # ... (other attributes)

        if i_size != -1:
            self.update_input(i_size)

    # ... (other methods)
```

-   `t` and `dt` are introduced to keep track of the time step and its step size for the Adam algorithm.
-   `B1` and `B2` represent the exponential decay rates for the first and second moment estimates respectively.
-   `eps` is a small constant added to avoid division by zero.

### Update Input Method:

```python
    def update_input(self, i_size):
        # ... (previous code)

        # Initialize first and second moment estimates for weights and biases
        self.m = np.zeros(self.W.shape)
        self.m0 = np.zeros(self.W0.shape)
        self.v = np.zeros(self.W.shape)
        self.v0 = np.zeros(self.W0.shape)

        # ... (rest of the method)
```

-   The `update_input` method is extended to initialize the first and second moment estimates (`m`, `m0`, `v`, and `v0`) for both weights and biases.

### Backpropagate Method (Updated with Adam Optimization):

```python
    def backpropagate(self, dLdA):
        # ... (previous code)

        # Update time step
        self.t += self.dt

        # Compute first and second moment estimates for weights and biases
        self.m = self.B1 * self.m + (1 - self.B1) * dLdW
        self.m0 = self.B1 * self.m0 + (1 - self.B1) * dLdW0

        self.v = self.B2 * self.v + (1 - self.B2) * dLdW ** 2
        self.v0 = self.B2 * self.v0 + (1 - self.B2) * dLdW0 ** 2

        # Bias-corrected moment estimates
        mhat = self.m / (1 - self.B1 ** self.t)
        mhat0 = self.m0 / (1 - self.B1 ** self.t)
        vhat = self.v / (1 - self.B2 ** self.t)
        vhat0 = self.v0 / (1 - self.B2 ** self.t)

        # Update weights and biases using Adam update rule
        self.W -= (self.eta / self.t ** 2 / np.sqrt(vhat + self.eps)) * mhat
        self.W0 -= (self.eta / self.t ** 2 / np.sqrt(vhat0 + self.eps)) * mhat0

        # ... (rest of the method)
```

-   The `backpropagate` method is enhanced to incorporate the Adam optimization algorithm.
-   The first and second moment estimates (`m`, `v`, `m0`, `v0`) are updated based on the gradients.
-   Bias-corrected moment estimates (`mhat`, `vhat`, `mhat0`, `vhat0`) are calculated to mitigate bias during early time steps.
-   The weights and biases are updated using the Adam update rule, which takes into account both the first and second moment estimates.

These modifications integrate the Adam optimization algorithm into the existing Dense neural network layer implementation, enhancing its training capabilities and convergence properties.
