
%% Lab 6.1: PCA 

clear all
close all

% load images 
% images size is 20x20. 

load('MNISTdata2.mat'); 

nrows=20;
ncols=20;

nimages = size(X,1);

%Show the images
% for I=1:40:nimages, 
%     imshow(reshape(X(I,:),nrows,ncols))
%     pause(0.1)
% end


%% Perform PCA following the instructions of the lab

Xn = normalize(X,'center','mean');

%calcular Sigma, U, A
Sigma = 1/(nimages-1)*(X'*X);
[U,A] = eig(Sigma);

%ordenar los vectores propios de U según los calores propios de A
[A,I] = sort(diag(A),'descend');
U = U(:, I);

%escoger valor de k
k_escogido = false;
k = 1;
while k_escogido == false && k <= (nrows*ncols)
    if((sum(A(1:k))/sum(A)) > 0.99)
        k_escogido = true;
    else
        k = k+1;
    end
end

%calcular Uk
Uk = U(:,(1:k));

%reducir dimensión de los datos estandarizados
Z = X*Uk;




%% Use the classifier from previous labs on the projected space







