function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

# Get Cost
# Create vector with h(x) for each row in X
hypo = sigmoid(X * theta);  
# Sum results using cost function for logistic regression
cost = sum( -y .* log(hypo) - (1 - y) .* log(1 - hypo) ) / m;
# Regularized sum should not include theta0 which is theta(1)
reg = (lambda / (2*m)) * sum(theta(2:length(theta)) .^ 2);
# Add in regularization sum to get a result that doesn't overfit
J = cost + reg;

# Now do the same for gradient, which is a vector the same size as theta (i.e. number of paramaters)
grad = (X' * (hypo - y)) ./ m;
# Create regularized sum, but skip theta(1)
for i = 2:length(theta)
  grad(i) = grad(i) + (lambda * theta(i) / m);
end
# Can this be vectorized?  YES! See lrCostFunction on ex3


% =============================================================

end
