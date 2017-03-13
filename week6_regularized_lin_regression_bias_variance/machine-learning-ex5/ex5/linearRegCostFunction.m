function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J_non_reg = (1 / (2*m)) * sum((X*theta - y).^2);

J_reg = (lambda / (2*m)) * sum(theta(2:end).^2);

J = J_non_reg + J_reg;

% partial derv for each feature                   
grad = 1/m .* transpose(X) * (X*theta - y);

grad_reg_term = lambda/m .* theta(2:end); % skip 1st
% add regularization term
grad(2:end) = grad(2:end) + grad_reg_term;




% =========================================================================
end
