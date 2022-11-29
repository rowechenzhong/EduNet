**fashion_mnist**
  -
  - Dsi700Dsi300Dso10-1669208993228816900 One pass through training dataset
  - Accuracy 0.8076
  - Average loss 0.5256595922541712

**Sinusoid**
  -
  - Dsi100Dre100Dsi100Dno1-1669211730538154200 100 passes through 10000 size training
  - Average loss 0.08742916


**BGD vs SGD**
  - 
  - Here's something I just learned -- small-batch BGD is *much* faster than SGD:
  - SGD:
    - send one input vector in
    - matrices
    - get output vector
    - calc dLdA
    - backprop
  - minibatch:
    - send an input matrix in (a collection of BATCH_SIZE inputs)
    - matrices
    - get output matrix
    - calc dLdA, *which is now a vector*
    - Do one backprop for all BATCH_SIZE inputs
  - This becomes like, a super obvious improvement once you realize CCEdLdA returns a one-hot vector, which we're now sending through matrices.
  - So calculating dLdA through summation contains like strictly more information