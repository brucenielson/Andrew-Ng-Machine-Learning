function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    #hypo = X * theta;
    #summed1 = sum((hypo - y));
    #summed2 = sum((hypo - y) .* X(:, 2));
    #theta(1) = theta(1) - alpha * (1/m) * summed1;
    #theta(2) = theta(2) - alpha * (1/m) * summed2;

    hypo = X * theta;
    for i = 1:size(X, 2);
      summed(i) = sum((hypo - y) .* X(:, i));
      theta(i) = theta(i) - alpha * (1/m) * summed(i);
    end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);


end

end
