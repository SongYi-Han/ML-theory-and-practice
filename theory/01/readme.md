# Linear regression 

### what is linear regression
* Linear regression is one of the core methods in supervised machine learning which takes the observed data as input-output
pairs. This means linear regression is the task of estimating the relationship between the inputs and the outputs and especially it assumes 
that this relationship is linear and estimates the relationship as a linear function. Such linear functions are called linear models and can be used to predict
the outputs for new inputs. This linear models can be computed by observed data using the least-squares estimation method.

### why use linear regression? 
* Many real-life problems can be approximated by linear models.
* Linear model can also be used for more complex problems that give rise to non-linear relationships, by using techniques like basis expansion which allow to transform nonlinear
relationships to linear ones.
* The least-squares estimation of the linear model allows closed form analytic solutions, which make the linear models relatively easy to obtain. This is an huge advantage
to the machine learning methods that rely on optimisation methods.

### Linear model 
* The linear model is the sum of the feature values weighted by the model parameters.
* Therefore, the model parameters are also called weights.
* Consider a data point (x, y). The linear model is of the form: 
`y = w0 + x1w1 + · · · + xDwD + ϵ`,
where x1, . . . ,xD are the entries in the feature vector x, y is the label paired with x, w0, . . . ,wD are the model parameters, and ϵ is the noise term. The parameter w0 is not paired
with any of the feature values. This parameter shifts the graph of the function, so it is called the bias term.
* The goal of linear regression is to learn the parameter vector w = (w0, w1, . . . ,wD)T from the training data. This is also called fitting or training a model.
* training model : We feed a machine learning algorithm of our choice with the training data, and the algorithm returns the parameter vector w that 
best estimates the relationship between the features and the label.
<p align="center">
<img width="500" alt="Screen Shot 2023-02-01 at 10 14 44 PM" src="https://user-images.githubusercontent.com/40763359/216165145-62779cdc-4456-4d6d-9391-7541389179a3.png">
</p>
* test model : Once we obtain the parameters w from the training phase, we can use the model to
predict the output, This is called the testing phase or deployment phase. 
<p align="center">
<img width="500" alt="Screen Shot 2023-02-01 at 10 15 19 PM" src="https://user-images.githubusercontent.com/40763359/216165165-f9746c09-53f8-4ff1-b636-50d9131d7114.png">
</p>

### Mean Squared Error Function
* The functions that measure the prediction errors are called loss functions or objective functions. A widely used loss function is the mean squared error (MSE) function.
* The prediction errors can be positive for some data points while negative for others. When we sum up the errors, they may cancel each other out.
To avoid this, we square the errors so they are always positive. Another way to avoid the cancellation problem is to apply, instead of the square function, the absolute function
* linear model ‚y = w0 + x · w1 that has the parameters w = (w0, w1), the MSE is defined as:
<p align="center">
<img width="527" alt="Screen Shot 2023-02-01 at 10 32 12 PM" src="https://user-images.githubusercontent.com/40763359/216168473-32dbc0d8-ccd1-410e-adea-61d8c004115d.png">
</p>

### Least-Squares Estimate for One-Dimensional Data
*  the least-squares estimate is a method help how to find the parameters that minimise the MSE loss function. 
*  The goal of the least-squares estimate is to minimise the MSE function with respect to the bias term w0 and the weight w1. To do this, we
first take the partial derivatives of L with respect to w0 and w1, and set the partial derivatives to 0.
* After rearranging terms, we get a system of equations, which are called normal equations:
<p align="center">
<img width="278" alt="Screen Shot 2023-02-01 at 10 39 30 PM" src="https://user-images.githubusercontent.com/40763359/216169431-e331f83f-4699-4175-a656-1df4ecbf0e8a.png">
</p>

* By rewriting the normal equations, we obtain the optimal solutions for w0 and w1:
<p align="center">
<img width="173" alt="Screen Shot 2023-02-01 at 10 40 19 PM" src="https://user-images.githubusercontent.com/40763359/216169620-c5ed4a1f-6e74-4ebe-bcc3-747cd78c1c84.png">
</p>


### Least-Squares Estimate for Multi-Dimensional Data
* we have a parameter vector with D+1 weights, for D > 1, we now derive the optimal parameter vector w for the linear model with D features : 
 <img width="440" alt="Screen Shot 2023-02-01 at 10 44 10 PM" src="https://user-images.githubusercontent.com/40763359/216170410-ab10f617-e374-4c61-8e8a-bdee92148dd7.png">

* rewriting the MSE function :
<img width="479" alt="Screen Shot 2023-02-01 at 10 44 25 PM" src="https://user-images.githubusercontent.com/40763359/216170443-95f70c44-ed42-4463-bb3e-30eea0dd51c7.png">

* we compute the partial derivatives of the MSE function with respect to every weight. Instead of showing the
equations for all partial derivatives, we can present the computation of the gradient
<img width="279" alt="Screen Shot 2023-02-01 at 10 47 43 PM" src="https://user-images.githubusercontent.com/40763359/216170959-67c0c682-62c4-4a9b-9c67-3ea343b42707.png">

* Assuming XTX is invertible, we solve for w to get the least-squares estimate for w:
<img width="184" alt="Screen Shot 2023-02-01 at 10 48 08 PM" src="https://user-images.githubusercontent.com/40763359/216170962-20bf5a8b-51f3-49c2-9e9e-871ddea92591.png">

* By replacing the w in the linear model by the least-squares estimate for w,
<img width="264" alt="Screen Shot 2023-02-01 at 10 48 12 PM" src="https://user-images.githubusercontent.com/40763359/216170963-2627bb00-db0d-4ceb-9d20-6e9dc681c141.png">

### Least Squares Estimate in the Presence of Outliers
* one downside of using MSE loss function is it is sensitiveto outliers.
* When the distance is squared in the MSE function, the already large outlier distance is blown up even more, which makes the outlier being heavily weighted
in the least-squares estimation.
* One way to solve the issue is to exclude the outlier from the data.
* Alternatively, one can choose a different loss function that is not very sensitive to
outliers, such as the mean absolute errors (MAE) function.

# Basis expansion

### what is the Basis expansion 
* To allow linear models to capture the non-linear relationships, we can use the basis expansion method. 
* The idea is to expand the data by adding new features that are non-linear to the original features in the data, 
so that the non-linear relationships in the original data become linear in the expanded data. Then, linear models can be trained on the expanded data to capture the relationships.
* A problem of the basis expansion approach is that it expands the data to have high dimensionalities, which leads to the high computational costs in the high-dimensional
spaces. 
* To avoid such high computational costs, we can use the kernel trick (e.g. polynomial kernel and the radial basis function kernel)

### Polynomial Basis Expansion for One-Dimensional Data
* To capture the quadratic function, we can use the linear model with the basis expansion function ψ: y = w · ψ(x),
where ψ expands the input value x to a vector ψ(x) and w is the parameter vector with the same dimension of ψ(x)
* A popular option for the basis expansion function is the polynomial basis expansion function:
<p align="center">
<img width="233" alt="Screen Shot 2023-02-02 at 12 05 43 AM" src="https://user-images.githubusercontent.com/40763359/216186367-4c54ae03-e0bf-4875-9b7e-e08dc7740d3a.png">
</p>

  * **Linear Model with Polynomial Degree 1** 
    * The polynomial basis expansion function with degree 1 has the form ψ1_poly(x) = [1, x]T 
    * This is the simple linear model which cannot capture the quadratic function
    

  * **Linear Model with Polynomial Degree 1** 
    * To capture the quadratic function better, we need features with higher degrees, such as x2 
    * The polynomial basis expansion function with degree 2 has the form ψ2_poly(x) = [1, x,x2]T
   
