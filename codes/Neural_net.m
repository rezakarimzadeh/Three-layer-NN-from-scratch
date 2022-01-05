clc;clear;close all
load('../hand_digit_data.mat')
%% show 100 samples of data
new_x = im2double(X);
rand_ind = randi([1, 5000], [1,100]);
data_sample = new_x(rand_ind,:);
for i=1:10
    for j=1:10
        images((i-1)*20+1:i*20,(j-1)*20+1:j*20 )= reshape(data_sample((i-1)*10+j,:),20,20);
    end
end
figure;imshow(images,[])
%%  split train and test data
train_data = zeros(3000,400);
test_data = zeros(2000,400);

train_label = zeros(3000,1);
test_label = zeros(2000,1);

for i = 1:10
    train_data((i-1)*300+1:i*300,:) = X((i-1)*500+1:i*500-200,:);
    test_data((i-1)*200+1:i*200,:) = X(i*500-200+1:i*500,:);
    
    train_label((i-1)*300+1:i*300,:) = y((i-1)*500+1:i*500-200,:);
    test_label((i-1)*200+1:i*200,:) = y(i*500-200+1:i*500,:);
end
onehot_train_label = (train_label==1:10);
onehot_test_label = (test_label==1:10);
%% show sample of train and test
im_num = 800;
figure;subplot(121);imshow(reshape(train_data(im_num,:),20,20));
title(['image label is: ', num2str(train_label(im_num))])
subplot(122);imshow(reshape(test_data(im_num,:),20,20));
title(['image label is: ', num2str(test_label(im_num))])
%% wieght definition
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;
ep =0.12;
initial_Theta1 = rand(hidden_layer_size, input_layer_size+1)*2*ep-ep;
initial_Theta2 = rand(num_labels, hidden_layer_size+1)*2*ep-ep;
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];  


costFunction = @(p) nnCostFunction(p, ...
input_layer_size, ...
hidden_layer_size, ...
num_labels, train_data, onehot_train_label, 2);
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% reshaping weigths
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
num_labels, (hidden_layer_size + 1));

%% test step
a1=[ones(size(test_data,1),1) test_data];
a2=sigmoid_val(a1*Theta1');
a3=[ones(size(a2,1),1) a2];
final=(a3*Theta2');
testdigits = sigmoid_val(final);
[row , col] = find(testdigits > 0.5);
out_net = zeros(size(test_label));
out_net(row) = col;
% accuracy
counter = 0;
for i=1:length(test_label)
    if out_net(i) == test_label(i)
        counter = counter+1;
    end
end
accuracy_pred = counter/length(test_label)
%% some output example
ind = 5;
figure;
for i =1:6
subplot(2,3,i);
ind = randi([1,2000],[1])
imshow(reshape(test_data(ind,:),20,20),[]);title(['prediction is: ', num2str(test_label(ind))])
end
%% show middle layer output
a1_feature=[1 test_data(ind,:)];
second_layer = zeros(100,100);

temp = 1;
for i =1:5
for j= 1:5
a2_feature=sigmoid_val(a1_feature.*Theta1(temp,:));
temp = temp +1;
second_layer((i-1)*20+1:i*20,(j-1)*20+1:j*20) = reshape(a2_feature(2:end),20,20);
end
end
figure;imshow(second_layer,[]);
title('middle layer output')






function val = sigmoid_val(x)
val = 1./(1+exp(-x));
end
