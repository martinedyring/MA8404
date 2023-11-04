# Deep Learning Based on Neural ODEs
Look at different activation functions
### Project in MA8404 - Numerical solution of time dependent differential equations

#### Task Description
**Deep learning based on neural ODEs**

In this project you shall study and implement a neural network for image classification based on neural ODEs.
Study the papers below, and implement the method. Use 2D point clouds as a test example. You are free to use any package or toolbox such as PyTorch or TensorFlow to do the implementation. Try to use different Explicit Runge-Kutta methods and compare results.

Study the following references.
* Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, pages 6572–6583, 2018.
* M. Benning, E. Celledoni, M. J. Ehrhardt, B. Owren and C.-B. Schönlieb, Deep learning as optimal control problems: Models and numerical methods, J. Comput. Dyn., 6 (2019), 171--198
* Giesecke, Elisa,  Kröner, Axel,  Classification with Runge-Kutta networks and feature space augmentation. J. Comput. Dyn.8(2021), no.4, 495–520.

Variants: b. Apply to different ODE models and different sigmas
-	$y’ = \sigma(Wy + b)$ (standard, $W$ is $m\times m$ and $b$ is $m$-vector)
-	$y’= U^T \sigma(Wy+b)$ where $U$ and $W$ are $p\times m$-matrices, $b$ a $p$-vector, and $y$ a $m$-vector
-	try $\sigma$ to be various different choices including RELU and $\tanh$.

--------------------------------------
### Step-by-step
First, I investigated the conventional ResNet (in `resnet.ipynb`). This implementation is quite complex w.r.t a given input x, and following it's tranformation thourout the network. However, this exercise was helpful to get an impression of how complicated the ResNet arcithecture can potentially lead into.

Next, I considered the RK-nets (EulerNet) (in `rk_net.ipynb`) similarly/inspired to those considered in the suggested literature. This was done to understand what is happening, without introducing any specific ODE to solve. Insted, I could focus in the dataflow.

At the end, I extended the RK-net into a NeuralOde (in `neural_ode.ipynb`). Here, I implemented the two suggested ODEs (y'), and used both a standard layer, forward euler and runge-kutta 4 layer to simulate a setp size though the network. This notebook the the final one, and are enriched with tranformation plots of the inputdata.