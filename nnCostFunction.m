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
X = [ones(m, 1) X];
         
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

%mult = ones(size(theta));
%mult(1) = 0;
%Theta1 size is 25   401
%Theta2 size is 10   26
%5000X401 401X25 -> 500X25 ->5000X26 -> 5000X26 X 26X10 -> 5000 X 10

H1 = 0;
H1 = sigmoid(X * Theta1');
H2 = 0;
H1 = [ones(m, 1), H1];
H2 = sigmoid(H1 * Theta2');% 5000 X 10
eye_matrix = eye(num_labels); %5000 X 10
y_matrix = eye_matrix(y,:);
%J = (-1 .* (y_matrix' * log(H2))- (-1 .* y_matrix' + 1) * log(-1.* H2 + 1))/m %+ sum((theta(2:end)).^2) * lambda /(2 * m);
%size(J)
J = 0;
for i=1:m
    for j=1:num_labels
    J = J - (y_matrix(i,j) * log(H2(i,j)))- (- y_matrix(i,j) + 1) * log(- H2(i,j) + 1);
    end
end
J=J/m;
%Theta(1,size(X,2)+1)
sums_theta_1 = 0;
sums_theta_2 = 0;
theta_reg = 0;

for i=1:hidden_layer_size %25  401
    for j=1:(size(X,2)-1)
        sums_theta_1 = sums_theta_1 + (Theta1(i,j+1))^2;
    end
end


for i=1:num_labels % 10 26
    for j=1:hidden_layer_size
        sums_theta_2 = sums_theta_2 + (Theta2(i,j+1))^2;
    end
end
theta_reg = (sums_theta_1 + sums_theta_2) * lambda / (2 * m);
J = J + theta_reg ;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. 


%After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
a1 = X;
n = size(X,2);
h = hidden_layer_size;
r = num_labels;

z2 = (X * Theta1');
a2 = [ones(m, 1), sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(a2 * Theta2');% 5000 X 10
%eye_matrix = eye(num_labels); %5000 X 10
%y_matrix = eye_matrix(y,:);

d3 = a3 - y_matrix;
d2 = (d3 * Theta2(:,2:end)).* sigmoidGradient(z2);% d2 is wRONG!!!!!d2 is same size as z2


D1 = (d2' * a1)./m;
D2 = (d3' * a2) ./m;
Theta1_grad = D1;
Theta2_grad = D2;


Theta_1_reg = Theta1;
Theta_1_reg(:,1) = 0;
Theta_2_reg = Theta2;
Theta_2_reg(:,1) = 0;

Theta1_grad = Theta1_grad + Theta_1_reg * lambda / m;
Theta2_grad = Theta2_grad + Theta_2_reg * lambda / m;


d2
d3
D1
D2
z2
a2
a3
sigmoidGradient(z2)

%size(Theta2_grad)
%size(Theta2)
%z2 = ones(hidden_layer_size);
%z3 = ones(num_labels);
%a2 = a1;
%a3 = a1;
%delta_3 = ones(num_labels);
%delta_2 = ones(hidden_layer_size);
%D_1 = ones(size(Theta1));%zeros(hidden_layer_size, size(X,2));
%D_2 = ones(size(Theta2));zeros(num_labels, hidden_layer_size);
%size(Theta1); % 25 X 401
%size(Theta2); % 10 X 26
%delta2vec = ones(m, hidden_layer_size);
%delta3vec = ones(m, num_labels);
%a2vec = ones(m, 1 + hidden_layer_size);
%a1vec = ones(size(X));

%for t=1:m
%    a1 = X(t,:);
%    z2 = a1 * Theta1';
%    a2= [1, sigmoid(z2)];
%    z3 = a2 * (Theta2)';
%    a3 = sigmoid(z3);
% backpropagation
%calc delta3
%    delta_3 = a3 - y_matrix(t,:);
  %calc delta2
 %   delta = (delta_3 * Theta2(:,2:end)).* sigmoidGradient(z2);
 %   delta_2 = delta;
    
  %  a1vec(t,:) = a1;
  %  a2vec(t,:) = a2;
  %  delta2vec(t,:) = delta_2;
  %  delta3vec(t,:) = delta_3;
    %(2:end);
   % size(a1); %1 X 401
   % size(delta_2);% 1 X 25
% D_1 = D_1 + delta_2' * a1; % D should be 25 X 401
   % size(a2); %1 X 26
   % size(delta_3); % 1 X 10
%  D_2 = D_2 + delta_3' * a2; % D2 should be 10 X 26
%end



%checkNNGradients(0);

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
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
