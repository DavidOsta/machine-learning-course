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
    % X(:,1) is always == 1
    
    % bsxfun(V,M) - to do elementwise multiplication of vector and matrix
    %   [1; 2; 3] * [1 1; 2 2; 3 3] = [1 1; 4 4; 9 9]
    % V = (h_theta * X - y)
    % M = X

    delta = (1/m) * sum(bsxfun(@times, X * theta - y, X));
    theta_temp = theta - alpha * transpose(delta);
    theta = theta_temp;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
