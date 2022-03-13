clear ; close all;
%% Load Data exams
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('exam_data.txt');
y = data(:, 3);
X = data(:, [1, 2]);
[Xtr, ytr, Xtest, ytest] = separar(X,y,0.8);

%% Regresión logística básica
Xtr = [ones(height(ytr),1) Xtr];
Xtest = [ones(height(ytest),1) Xtest];
theta_ini = [0 0 0]';

options = [];
options.display = 'final'; %otros: 'iter' , 'none‘
options.method = 'newton'; %por defecto: 'lbfgs'

%entrenamiento
theta = minFunc(@costeLogistico,theta_ini, options, Xtr, ytr);

%tasa de error entrenamiento
h = 1./(1+exp(-(Xtr*theta)));
ytr_pred = double(h >= 0.5);
Etr = tasa_error(ytr_pred,ytr);

%tasa de error test
h = 1./(1+exp(-(Xtest*theta)));
ytest_pred = double(h >= 0.5);
Etest = tasa_error(ytest_pred,ytest);

%dibujar recta de regresión logística
plotDecisionBoundary(theta, Xtr, ytr);
xlabel('Exam 1 score');
ylabel('Exam 2 score');

%% Ejemplo alumno
segExamen = 1:100;
size(segExamen)
size(ones(100,1))
examen = [ones(100,1) ones(100,1)*45 segExamen'];
h = 1./(1+exp(-(examen*theta)));

figure;
grid on; hold on;
xlabel('Nota 2º examen');
ylabel('Probabilidad de admisión');
plot(examen(:,3), h, 'r-');

%% Load data mchip

data = load('mchip_data.txt');
y = data(:, 3);
X = data(:, [1, 2]);
N = length(y);
[Xtr, ytr, Xtest, ytest] = separar(X,y,0.8);

%% Regularización

Xtr_exp = mapFeature(Xtr(:,1),Xtr(:,2));    %expansión de funciones base
lambda = logspace(-6,5);    %vector de lambdas logarítimico

%elegir mejor parámetro de regularización
[best_lambda,Etr,Ecv] = kfold_cross_validation(Xtr_exp,ytr,10,lambda')

%Dibujar evolución tasas de error de entrenamiento y validación
figure;
grid on; hold on;
ylabel('Tasa de error'); xlabel('Factor de regularización');

set(gca, 'XScale', 'log', 'YScale', 'log');
loglog(lambda, Etr, 'r-');
loglog(lambda, Ecv, 'b-');
loglog(lambda(best_lambda), Ecv(best_lambda), 'g.');
legend('Tasa de error de entrenamiento', 'Tasa de error de validación', 'Punto ideal');

%entrenar todos los datos mejor modelo
H = Xtr_exp'*Xtr_exp + lambda(best_lambda)*diag([0 ones(1,size(Xtr_exp,2)-1)]);
th1 = H \ (Xtr_exp'*ytr);

%entrenar todos los datos lambda = 0
H = Xtr_exp'*Xtr_exp + 0*diag([0 ones(1,size(Xtr_exp,2)-1)]);
th2 = H \ (Xtr_exp'*ytr);

%dibujar recta de regresión logística
plotDecisionBoundary(th1, Xtr, ytr);
xlabel('Exam 1 score');
ylabel('Exam 2 score');