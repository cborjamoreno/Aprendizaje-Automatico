clear ; close all;
addpath(genpath('../minfunc'));

%% Load Data MNIST
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
[Xtr, ytr, Xcv, ycv] = separar(X,y,0.8); %   Separar 20% de los datos para 
                                         % validación
N_pixels = width(Xtr);
N_clases = 10;

%% Regresión logística regularizada
Xtr = [ones(height(ytr),1) Xtr];
Xcv = [ones(height(ycv),1) Xcv];

options = [];
options.display = 'final';
options.method = 'lbfgs';
lambda = logspace(-6, 2);

Etr = [];
Ecv = [];

best_model = 0;
best_errV = inf;

for model = 1:length(lambda)
    etr = 0; ecv = 0;
    theta = zeros(N_pixels+1,1);
    for i = 1:10
        y_clasif = (ytr == i);  %   Obtener salidas booleanas para cada
                                % clasificador
        th = minFunc(@costeLogReg, zeros(N_pixels+1,1), options, Xtr, ...
            y_clasif, lambda(model));
        theta = [theta th];
    end

    ytr_pred = pred_sigmoid(Xtr,theta);
    etr = tasa_error(ytr_pred,ytr);

    ycv_pred = pred_sigmoid(Xcv,theta);
    ecv = tasa_error(ycv_pred,ycv);

    Etr = [Etr;etr];
    Ecv = [Ecv;ecv];

    if ecv < best_errV
        best_errV = ecv;
        best_model = model;
    end
end

%Dibujar evolución tasas de error de entrenamiento y validación
figure;
grid on; hold on;
ylabel('Tasa de error'); xlabel('Factor de regularización');

plot(log10(lambda), Etr, 'r-', 'LineWidth',1);
plot(log10(lambda), Ecv, 'b-', 'LineWidth',1);
plot(log10(lambda(best_model)), Ecv(best_model), 'g.');
legend('Tasa de error de entrenamiento', 'Tasa de error de validación', 'Punto ideal');

%% Matriz de confusión y Precisión/Recall

% Entrenar todos los datos con el mejor modelo
Xtest = [ones(height(ytest),1) Xtest];
X = [ones(height(y),1) X];
theta = zeros(N_pixels+1,1);

for i = 1:10
    y_clasif = (y == i);
    th = minFunc(@costeLogReg, zeros(N_pixels+1,1), options, X, ...
        y_clasif, lambda(best_model));
    theta = [theta th];
end

ytest_pred = pred_sigmoid(Xtest,theta);

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
