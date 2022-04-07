clear ; close all;

%% Load Data MNIST
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
[Xtr, ytr, Xcv, ycv] = separar(X,y,0.8); %   Separar 20% de los datos para 
                                         % validación
N_pixels = width(Xtr);
N_clases = 10;
[N_tr,~] = size(ytr);

[unos,~] = size(find(ytr==1));

Xi = Xtr((ytr==1),:);
mu_i = mean(Xi);
cov(Xi);