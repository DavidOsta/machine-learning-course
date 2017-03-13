function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 10;
% sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];

lowest_error = intmax;

C = 0;
sigma = 0;

for c = C_vec
    for s = sigma_vec
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s)); 
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if(error < lowest_error)
            lowest_error = error;
            C = c;
            sigma = s;
        end
    end
end

% =========================================================================

end
