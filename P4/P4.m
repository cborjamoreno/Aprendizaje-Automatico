clear ; close all;
addpath(genpath('../minfunc'));

%% Load Data
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);

% Inventa una solucion y muestra las confusiones
yhat = ceil(10*rand(size(y)));
verConfusiones(X, y, yhat);
