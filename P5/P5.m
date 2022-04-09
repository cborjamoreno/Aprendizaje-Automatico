clear ; close all;

%% Load Data MNIST
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
[Xtr, ytr, Xcv, ycv] = separar(X,y,0.8); %   Separar 20% de los datos para 
                                         % validación
N_pixels = width(Xtr);
N_clases = 10;

%% Bayes ingenuo
lambda = logspace(-6, 2);

[Etr,Ecv,best_lambda] = entrenarYclasificarBayes(Xtr,ytr,Xcv,ycv,N_clases, ...
    lambda,1);

dibujarEvolucionErrores(lambda,best_lambda,Etr,Ecv);

disp(lambda(best_lambda));
%% Matriz de confusión y Precisión/Recall para el modelo de Bayes ingenuo

% Entrenar todos los datos con el mejor modelo

modelo = entrenarGaussianas(X,y,N_clases,1,lambda(best_lambda));
ytest_pred = clasificacionBayesiana(modelo,Xtest);

% Calcular matriz de confusión

m = zeros(N_clases,N_clases);

for i=1:N_clases
    for j=1:N_clases
        m(i,j) = sum(ytest_pred == i & ytest == j);
    end
end


%   Array de precisiones. La componente i in [1,10] contiene la precisión
% de la clase i. La precisión de la clase 0 está en la componente i=10.
P = zeros(1,N_clases);

%   Array de recalls. La componente i in [1,10]contiene el recall de la 
% clase i. El recall de la clase 0 está en la componente i=10.
R = zeros(1,N_clases);

% Cálculo precisión de cada clase
for i=1:N_clases-1
    P(i) = m(i,i)/sum(m(i,:));
end
P(10) = m(10,10)/sum(m(10,:));

% Cálculo recall de cada clase
for i=1:N_clases-1
    R(i) = m(i,i)/sum(m(:,i));
end
R(10) = m(10,10)/sum(m(:,10));

disp(m);
disp(P);
disp(R);

verConfusiones(Xtest, ytest, ytest_pred);

%% Covarianzas completas

[Etr,Ecv,best_lambda] = entrenarYclasificarBayes(Xtr,ytr,Xcv,ycv,N_clases, ...
    lambda,0);

dibujarEvolucionErrores(lambda,best_lambda,Etr,Ecv);

disp(lambda(best_lambda));

%% Matriz de confusión y Precisión/Recall para el modelo de Covarianzas Completas

% Entrenar todos los datos con el mejor modelo

modelo = entrenarGaussianas(X,y,N_clases,1,lambda(best_lambda));
ytest_pred = clasificacionBayesiana(modelo,Xtest);

% Calcular matriz de confusión

m = zeros(N_clases,N_clases);

for i=1:N_clases
    for j=1:N_clases
        m(i,j) = sum(ytest_pred == i & ytest == j);
    end
end


%   Array de precisiones. La componente i in [1,10] contiene la precisión
% de la clase i. La precisión de la clase 0 está en la componente i=10.
P = zeros(1,N_clases);

%   Array de recalls. La componente i in [1,10]contiene el recall de la 
% clase i. El recall de la clase 0 está en la componente i=10.
R = zeros(1,N_clases);

% Cálculo precisión de cada clase
for i=1:N_clases-1
    P(i) = m(i,i)/sum(m(i,:));
end
P(10) = m(10,10)/sum(m(10,:));

% Cálculo recall de cada clase
for i=1:N_clases-1
    R(i) = m(i,i)/sum(m(:,i));
end
R(10) = m(10,10)/sum(m(:,10));

disp(m);
disp(P);
disp(R);

verConfusiones(Xtest, ytest, ytest_pred);