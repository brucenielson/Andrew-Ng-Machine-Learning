function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

# −y(i) log(hθ(x(i)))−(1−y(i))log(1−hθ(x(i))) 



# Get Cost
# Create vector with h(x) for each row in X
hypo = sigmoid(X * theta);  
# Sum results using cost function for logistic regression
J = sum( -y .* log(hypo) - (1 - y) .* log(1 - hypo) ) / m;


# Gradient for cost function
#for i = 1:size(X, 2); # Run through each theta in a vector of thetas
#  grad(i) = sum((hypo - y) .* X(:, i)) / m;
#end

# Vectorized version
grad = (X' * (hypo - y)) ./ m;





% =============================================================

end
