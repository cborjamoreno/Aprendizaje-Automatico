%% Based on exercise 2 of Machine Learning Online Class by Andrew Ng 

clear ; close all;

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('exam_data.txt');
y = data(:, 3);
X = data(:, [1, 2]);
N = length(y);
[Xtr, ytr, Xtest, ytest] = separar(X,y,0.8);

plotData(Xtr, ytr);
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')


%% Regresión logística básica
Xtr = [ones(height(ytr),1) Xtr];
Xtest = [ones(height(ytest),1) Xtest];
theta_ini = [0 0 0]';

options = [];
options.display = 'final'; %otros: 'iter' , 'none‘
options.method = 'newton'; %por defecto: 'lbfgs'

%entrenamiento
theta = minFunc(@costeLogistico,theta_ini, options, Xtr, ytr)

%tasa de error entrenamiento
h = 1./(1+exp(-(Xtr*theta)));
ytr_pred = double(h >= 0.5);
Etr = tasa_error(ytr_pred,ytr)

%tasa de error test
h = 1./(1+exp(-(Xtest*theta)));
ytest_pred = double(h >= 0.5);
Etest = tasa_error(ytest_pred,ytest)

%dibujar recta de regresión logística
plotDecisionBoundary(theta, Xtr, ytr);
xlabel('Exam 1 score');
ylabel('Exam 2 score');

%% Ejemplo alumno
examen = [ones(100,1)*45 0:100];
h = 1./(1+exp(-(examen*theta)));
plot(examen,, 'r-');
