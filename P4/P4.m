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
        y_clasif = (ytr == i);
        th = minFunc(@costeLogReg, zeros(N_pixels+1,1), options, Xtr, ...
            y_clasif, lambda(model));
        theta = [theta th];
    end

    h = 1./(1+exp(-(Xtr*theta)));
    h(:,1) = [];
    [~,h] = max(h,[],2);

    etr = tasa_error(h,ytr);

    h = 1./(1+exp(-(Xcv*theta)));
    h(:,1) = [];
    [~,h] = max(h,[],2);

    ecv = tasa_error(h,ycv);

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

h = 1./(1+exp(-(Xtest*theta)));
h(:,1) = [];
[~,ytest_pred] = max(h,[],2);

% Calcular matriz de confusión

m = zeros(N_clases,N_clases);

for i=1:N_clases
    for j=1:N_clases
        m(i,j) = sum(ytest_pred == i & ytest == j);
    end
end

disp(m);

% TP = sum(ytest_pred == 1 & ytest == 1);
% FP = sum(ytest_pred == 1 & ytest == 0);
% TN = sum(ytest_pred == 0 & ytest == 0);
% FN = sum(ytest_pred == 0 & ytest == 1);
% 
% P = TP/(TP+FP);
% R = TP/(TP+FN);
% 
% M_CONF = [TP FP; TN FN];
% disp(M_CONF);

verConfusiones(Xtest, ytest, ytest_pred);
