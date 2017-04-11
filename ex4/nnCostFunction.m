function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

# Implementing: \begin{gather*}\large J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}

#size(X) = 5000 x 400 is the Training Examples with 5000 examples and 400 features (i.e. 20 x 20 pixels) I will add bias node later
#size(Theta1) = 25 x 401 is the Theta parameters for the first hidden layer is, for each example, 400 features (plus bias) by 25 nodes
#size(Theta2) = 10 x 26 is the Theta parameters for the output layer, with 25 inputs (plus bias) going to 10 output nodes
#size(y) = 5000 x 1 is the correct answer (listed as 1 to 10) for all 5000 examples

#Other numbers
#K = output units = 10 = num_labels
#m = # of examples
#i = iteration through examples
#k = iteration through output units

#Recode y to instead of having a correct label (i.e. 1 to 10) it is encoded as T/F codes i.e. [0;0;0;1;0...]
id = eye(num_labels); # All results in Y sholud contain one of these identities Size 10 x 10
Y =  id(y,:);# Set Y (list of outputs) to all zeros over all examples. Size: 5000 x 10
# same as
#for i=1:m
#  Y(i, :) = id(y(i), :);
#end


# Now, for each example, sum over all output units using essentially a slightly modified logistic regression formula
# Create vector with h(x) for each row in X (i.e. 1 to m)
a1 = [ones(m, 1) X]; # Size 5000 x 401 is X plus bias node 
z2 = a1 * Theta1'; # Size: 5000 x 401 * 401 x 25 = 5000 x 25
a2 = sigmoid(z2); # Size: 5000 x 25
a2 = [ones(m, 1) a2]; # Size: 5000 x 26 (added bias node)
z3 = a2 * Theta2'; # Size: 5000 x 26 * 26 x 10 = 5000 x 10
a3 = sigmoid(z3);


# Get Cost
# Sum results using cost function for logistic regression
# The thing I was getting wrong was I was confusing y with Y and you have to sum Y and hypo together at the same time
# This ultimately requires a double sum
# Y and hypo are both 5000 x 10, so you can do a dot operator on them
cost = sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3))) / m;


# Create Regularized Sum
# Regularized sum should not include the bias node
reg = (lambda / (2*m)) * ( sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)) );
# Add in regularization sum to get a result that doesn't overfit
J = cost + reg;

# Back propagation
d3 = a3 - Y; # Size: 5000 x 10
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2); # Size: 5000 x 10 * 10 x 25 = 5000 x 25
Delta1 = d2' * a1; # Size: 25 x 5000 * 5000 x 401 = 25 x 401 (hidden units x features)
Delta2 = d3' * a2; # Size: 10 x 5000 * 5000 x 26 = 10 x 26
Theta1_grad = Delta1 / m; # Size: 15 x 401
Theta2_grad = Delta2 / m; # Size: 10 x 26

# Regularization of Neural Net
# Set first column of Theta1 and Theta2 to all zeros
Theta1(:,1) = 0;
Theta2(:,1) = 0; 
# Scale using regularization
Theta1 = Theta1 .* (lambda/m);
Theta2 = Theta2 .* (lambda/m);
# Add regularization
Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
