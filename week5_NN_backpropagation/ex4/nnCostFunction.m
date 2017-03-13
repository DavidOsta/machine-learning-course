function [J, grad] = nnCostFunction(nn_params, ...
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

%% Part 1

bias_ones = ones(m, 1);
a1 = [bias_ones X]; % input with 1 bias

z2 = a1 * Theta1';
a2 = [bias_ones sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3); % a3==hypothesis => no need for biased unit

h_theta = a3;

y_vector = zeros(m, num_labels);

for k=1:num_labels
   y_vector(:,k) = (y == k);
end
% for-cycle implementation
% J = 0;
% for k=1:m
%   J = J + sum(-y_vector(k,:) .* log(h_theta(k,:)) - (1 - y_vector(k,:)) .* log(1 - (h_theta(k,:))))
% end

% J = J / m;

J_non_reg = 1/m * sum(sum(-1 * y_vector .* log(h_theta) - (1 - y_vector) .* log(1 - h_theta)));
% do not regularize BIAS
J_reg = (lambda / (2 * m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J_non_reg + J_reg;

%% Part 2

Delta1 = 0;
Delta2 = 0;

for k = 1 : m
    % try to convert a1 to column vector first
    % feedforward
    a1_t = a1(k,:);
    z2_t = a1_t * Theta1';
    a2_t = [1 sigmoid(z2_t)];
    z3_t = a2_t * Theta2';
    a3_t = sigmoid(z3_t); 

    % errors
    delta3 = transpose(a3_t - y_vector(k,:));
    delta2 = (Theta2' * delta3) .* sigmoidGradient([1 z2_t]');
    
    % Accumulate gradient
    Delta1 = Delta1 + delta2(2:end) * a1_t;
    Delta2 = Delta2 + delta3 * a2_t; % output == there is no Bias unit


end

Theta1_grad_non_reg = 1/m * Delta1;
Theta2_grad_non_reg = 1/m * Delta2;

% ignore bias units
Theta1_grad_reg = Theta1_grad_non_reg(:,2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad_reg = Theta2_grad_non_reg(:,2:end) + lambda / m * Theta2(:, 2:end);

% =========================================================================

% Unroll gradients
Theta1_grad = [Theta1_grad_non_reg(:,1) , Theta1_grad_reg];
Theta2_grad = [Theta2_grad_non_reg(:,1) , Theta2_grad_reg];

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
