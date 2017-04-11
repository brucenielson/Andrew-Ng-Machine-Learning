function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

# X is # Movies X # Features
# Theta is # of Users X # of Features (Actually # Thetas corresponding to features)size(Y)
# Y is # Movies X # Users
# R is # Movies X # Users

# Combine Y and R together. 
Y = Y .* R;

# Create predictions. Create a combined X and Theta matrix. This will work out to be a # Movies X # Users matrix of same size as Y.
predict = X * Theta';
# For a cost function, we are only interested in comparing rating predictions with actual ratings made
# So we will remove all where the R matrix doesn't equal 1.
predict = predict .* R;

# Distance for cost purposes. Note: This is the error factor to use to calc gradients. It is # Movies X # Users
diff = (predict - Y);
dist = diff .^ 2;

# Final Cost
J = sum(sum(dist)) / 2;


# Regularize the Cost
reg = lambda * ( sum(sum(Theta .^2)) + sum(sum(X .^2)) ) / 2;
J = J + reg;



# Calculate Gradients: Error Factor' * Theta gives # Movies X # Users * # Users X # Features = # Movies X # Features
X_grad = diff * Theta;
# Same as above, but use X instead of Theta
# I.E. Error Factor' X X gives # Users X # Movies * X # Movies X # Features = # Users X # Features
Theta_grad = diff' * X;

# Regularize the gradient
reg_X = X * lambda; # # Movies X # Features
reg_Theta = Theta * lambda; # # Users X # Features

X_grad = X_grad + reg_X;
Theta_grad = Theta_grad + reg_Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
