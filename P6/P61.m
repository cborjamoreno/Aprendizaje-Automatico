
%% Lab 6.1: PCA 

clear all
close all

% load images 
% images size is 20x20. 

load('MNISTdata2.mat'); 

nrows=20;
ncols=20;

nimages = size(X,1);
N_clases = 10;

%Show the images
% for I=1:40:nimages, 
%     imshow(reshape(X(I,:),nrows,ncols))
%     pause(0.1)
% end


%% Perform PCA following the instructions of the lab

%estandarizar datos
Xn = normalize(X,'center','mean');

%calcular Sigma, U, A
Sigma = 1/(nimages-1)*(Xn'*Xn);
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
Z = Xn*Uk;

%% Use the classifier from previous labs on the projected space

%entrenar con datos de entrenamiento reducidos
[Ztr, ytr, Zcv, ycv] = separar(Z,y,0.8); %   Separar 20% de los datos para 
                                         % validación
lambda = logspace(-6, 2);
[Etr,Ecv,best_lambda] = entrenarYclasificarBayes(Ztr,ytr,Zcv,ycv,N_clases, ...
    lambda,0);

disp(lambda(best_lambda));

dibujarEvolucionErrores(lambda,best_lambda,Etr,Ecv);

%obtener Z de los datos de test
Xtestn = normalize(Xtest,'center','mean');
Ztest = Xtestn*Uk;

modelo = entrenarGaussianas(Z,y,N_clases,0,lambda(best_lambda));
ytest_pred = clasificacionBayesiana(modelo,Ztest);

verConfusiones(Xtest, ytest, ytest_pred);



