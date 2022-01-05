% input_layer_size = 400;
% hidden_layer_size = 25;
% num_labels = 10;
% ep =0.12;
% initial_Theta1 = rand(hidden_layer_size, input_layer_size+1)*2*ep-ep;
% initial_Theta2 = rand(num_labels, hidden_layer_size+1)*2*ep-ep;
% initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];  

function [J grad] = nnCostFunction(initial_nn_params, ...
input_layer_size, ...
hidden_layer_size, ...
num_labels, ...
train_data,onehot_train_label, lambda)

theta1 = reshape(initial_nn_params(1:hidden_layer_size*(input_layer_size+1)),hidden_layer_size, input_layer_size+1);
theta2 = reshape(initial_nn_params(hidden_layer_size*(input_layer_size+1)+1:end),num_labels,hidden_layer_size+1);
%% forward path
a1=[ones(size(train_data,1),1) train_data];
z2 = a1*theta1';
a2 = sigmoid_val(z2);
a3=[ones(size(a2,1),1) a2];
z3 = a3*theta2';
a4 = sigmoid_val(z3);

m = size(train_data,1);

cost_mat = -onehot_train_label.*log(a4)-(1-onehot_train_label).*log(1-a4);

vec_theta1 = theta1(:,2:end);
vec_theta1 = vec_theta1(:);
vec_theta2 = theta2(:,2:end);
vec_theta2 = vec_theta2(:);
J = sum(cost_mat(:))/m+(lambda/(2*m)) * (sum(vec_theta1.^2)+sum(vec_theta2.^2));
%% backpropagation
D1 =0; D2=0;
dj_dtheta1 = zeros(size(theta1));
dj_dtheta2 = zeros(size(theta2));
for i = 1:m
    xx = train_data(i,:);
    a1=[1 xx];
    z2 = a1*theta1';
    a2 = sigmoid_val(z2);
    a3=[1 a2];
    z3 = a3*theta2';
    a4 = sigmoid_val(z3);
    
    d1 = (a4 - onehot_train_label(i,:))';
    d2 = (theta2'*d1).*sigmoid_der([1 z2])';
    d2 = d2(2:end);
    
    D1 = D1 + d1*a3;
    D2 = D2 + d2*a1;
end
D1 = D1/m;
D2 = D2/m;

dj_dtheta2(:,2:end) = D1(:,2:end) +(lambda/m)*theta2(:,2:end) ;
dj_dtheta2(:,1) = D1(:,1);

dj_dtheta1(:,2:end) = D2(:,2:end)+(lambda/m)*theta1(:,2:end) ;
dj_dtheta1(:,1) = D2(:,1);

grad = [dj_dtheta1(:) ; dj_dtheta2(:)];

end

function val = sigmoid_val(x)
val = 1./(1+exp(-x));
end

function der = sigmoid_der(x)
der = sigmoid_val(x).*(1-sigmoid_val(x));
end