# Developing Backpropagation from Scratch

In this work, I coded a feedforward neural network from scratch. I have used the gradient descent method and its variants such as `mini_batch_gd`, `momentum_gd`, `nesterov_gd`, `rmsprop`, `adam`, and `nadam`, along with weight initializations such as `Xavier` and `Random`. I also utilized various activation functions including `sigmoid`, `tanh`, `relu`, and `softmax`, as well as loss functions such as `MSE` and `CrossEntropy`. These techniques were applied with backpropagation to classify images from the Fashion-MNIST dataset. I used `wandb.ai` to perform experiments for hyperparameter tuning.

## Libraries and Their Application

1. **Numpy:** For mathematical operations.
2. **Keras:** To obtain the dataset.
3. **Matplotlib:** For plotting the confusion matrix.
4. **Seaborn:** For plotting sample images from each class.
5. **Sklearn:** To split the dataset into train, test, and validation sets.
6. **Wandb:** To log the metrics to wandb.ai.

## Installations

The above-mentioned libraries can be installed on the local machine by using the following code snippet in the command prompt:
```python
pip install numpy
pip install keras
pip install matplotlib
pip install seaborn
pip install sklearn
pip install wandb
```
If you are running the code on Google Colab, all the above-mentioned libraries are already installed **except** "wandb". Add the following code in a cell:
```python
!pip install wandb
```

## Training the Neural Network

To train the neural network, use the following function:
```python
fit(X_train, 
    y_train,
    layer_sizes,
    wandb_log, 
    learning_rate=0.0001, 
    initialization_type="random", 
    activation_function="sigmoid", 
    loss_function="cross_entropy", 
    mini_batch_size=32, 
    max_epochs=5, 
    lambd=0,
    optimization_function=mini_batch_gd)
```
 1. `X_train` stores the list of flattened images of the training dataset.
  2. `y_train` stores the list of labels for the images of the training dataset (one-hot encoded format).
  3. `layer_sizes` stores  the number of neurons present in each layer (Both the input and the output layers).
  4. `wandb_log` stores the boolean variable which determines whether or not the data is logged into wandb.ai
  5. `learning_rate` stores the learning rate of the gradient descent(`mini_batch_gd` `momentum_gd`, `nesterov_gd`, `rmsprop`, `adam`, `nadam` ) optimization functions
  6. `initialization_type` stores the weight initialization type (`Xavier` or `random`)
  7. `activation_function` stores the activation function that is applied to all the hidden layers
  8. `loss_function` stores the type of loss function (`cross_entropy` or `squared error`)
  9. `mini_batch_size` stores the number of data points per batch.
  10. `max_epochs` stores the maximum number of epochs 
  11. `lambd` stores the regularization constant for weight decay
  12. `optimization_function` stores the name of the gradient descent algorithm

## Template for Adding a New Optimization Function

We have provided a template for adding an optimization function on similar lines to previous functions. The user needs to add the following code snippets to form a new optimization function:
1. Declare and initialize dictionaries and other data structures as per the requirement of the optimization function.
2. New parameter update rule for the network parameters.

The new optimization function looks like this:
```python
new_optimization_function_name(X_train,
                              y_train,
                              eta,
                              max_epochs,
                              layers,
                              mini_batch_size,
                              lambd,
                              loss_function,
                              activation,
                              parameters,
                              wandb_log=False)
```
 1. `X_train` stores the list of flattened images of the training dataset.
  2. `y_train` stores the list of labels for the images of the training dataset (one-hot encoded format).
  3. `eta` stores the learning rate.
  4. `max_epochs` stores the maximum number of epochs.
  5. `layers` stores the number of neurons per each layer.
  6. `mini_batch_size` stores the number of data points per batch.
  7. `lambd` stores the regularization constant for weight decay.
  8. `loss_function` stores the type of loss function (`cross_entropy` or `squared error`).
  9. `activation` stores the activation function that is applied to all the hidden layers.
  10. `parameters` stores the intial parameters (`weights` and `biases`).
  11. `wandb_log` stores the boolean variable which determines whether or not the data is logged into wandb.ai
## Wandb Functionality

1. **To use wandb mode:** Find your API key from your wandb account and paste it in the output box after you execute this code snippet:
   ```python
   !wandb login --relogin
   # Enter the entity and project name in these variables
   entity_name="_entity_name_"
   project_name="_project_name_"
   ```
2. **Perform Experiments:**  
   Run sweeps using this function:
   ```python
   sweeper(entity_name, project_name)
   ```

3. **Compare Loss Functions:**  
   Compare the performance of two loss functions using this function:
   ```python
   loss_compare_sweeper(entity_name, project_name)
   ```

4. **Plot the Confusion Matrix:**  
   Use this function to plot the confusion matrix for the test dataset and get predicted and true labels:
   ```python
   y_pred, y_t = plot_confmat_wandb(entity_name, project_name)
   ```

5. **Resume a Sweep:**  
   Use the following command to resume a paused sweep:
   ```python
   !wandb sweep --resume wandb_user_name/project_name/sweep_ID
   ```

6. **Stop a Sweep:**  
   Use the following command to stop an ongoing sweep:
   ```python
   !wandb sweep --stop wandb_user_name/project_name/sweep_ID
   ```

7. **Cancel a Sweep:**  
   Use the following command to cancel a sweep:
   ```python
   !wandb sweep --cancel wandb_user_name/project_name/sweep_ID
   ```

8. **Run an Agent:**  
   Use the following command to run a wandb agent for the specified sweep:
   ```python
   wandb.agent("sweep_ID", project="project_name", function=train)
   ```

## Available Options to Customize the Neural Network

### 1) Weight Initializations

```python
Xavier()
Random()
```
### 2) Activation Functions

```python
sigmoid()
tanh()
relu()
softmax()
```

### 3) Optimization Functions

```python
mini_batch_gd()
momentum_gd()
nesterov_gd()
rmsprop()
adam()
nadam()
```

### 4) Loss Functions

```python
MSE()
CrossEntropy()
```

## References
- **Lecture Slides:** Prof. Mitesh Khapra's course CS6910 - Fundamentals of Deep Learning.  
  - [CS6910 - Fundamentals of Deep Learning](http://www.cse.iitm.ac.in/~miteshk/CS6910.html)
- **YouTube Lectures:** Prof. Mitesh Khapra's DeepLearning course lectures on deep learning fundamentals.  
  - [DeepLearning course lectures](https://www.youtube.com/playlist?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- **Official Documentation:** Python NumPy and wandb.ai.  
  - [Python Documentation](https://docs.python.org/3/)
  - [NumPy Documentation](https://numpy.org/doc/)
  - [wandb.ai Documentation](https://docs.wandb.ai/tutorials)
- **GitHub Repositories:** Open-source code repositories relevant to deep learning and neural networks.  
  - [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
  - [Awesome Artificial Intelligence](https://github.com/owainlewis/awesome-artificial-intelligence)
- **Academic Research:** Papers from arXiv/academic journals that provide theoretical insights and recent advancements in deep learning.
  1. **"Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014)**
     - **Link:** [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
  2. **"On the Convergence of Adam and Beyond" by Reddi, Kale, and Kumar (2018)**
     - **Link:** [arXiv:1904.09237](https://arxiv.org/abs/1904.09237)
  3. **"Averaged Stochastic Gradient Descent with Weight Dropped Convergence Rate" by Junchi Li, Fadime Sener, and Vladlen Koltun (2021)**
     - **Link:** [arXiv:2106.01409](https://arxiv.org/abs/2106.01409)
- **Online Forums:** Reddit's r/MachineLearning and r/deeplearning for discussions and knowledge sharing.  
  - [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
  - [r/deeplearning](https://www.reddit.com/r/deeplearning/)
- **Coursera Courses:** Andrew Ng's ML Specialization and DL Specialization on Coursera.  
  - [Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/)
  - [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/)
- **Additional Resources:**
  - [Optimization in Deep Learning: AdaGrad, RMSProp, Adam](https://artemoppermann.com/optimization-in-deep-learning-adagrad-rmsprop-adam/)
  - [Difference between RMSprop with momentum and Adam optimizers](https://datascience.stackexchange.com/questions/26792/difference-between-rmsprop-with-momentum-and-adam-optimizers)
  - [Optimization Techniques in Deep Learning](https://blogs.brain-mentors.com/optimization-techniques-in-deep-learning/)
  - [An overview of gradient descent optimization algorithms by Sebastian Ruder](https://www.ruder.io/optimizing-gradient-descent/)
---
