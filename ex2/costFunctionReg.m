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


g= sigmoid(X*theta);
thetaEx = theta(2:length(theta),1);
J = (1/m)*sum(-y .* log(g) - (1-y).*log(1-g)) + (lambda/(2*m))* sum(thetaEx.^2);

grad1 = (1/m)*((g-y)'*X)' ;
grad2 = (lambda/m).* thetaEx;

grad(1) = grad1(1);
grad(2:length(grad)) = grad1(2:length(grad)) + grad2;
% =============================================================

end
