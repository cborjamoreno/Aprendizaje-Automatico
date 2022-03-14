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
ytr_pred = 1./(1+exp(-(Xtr*theta)))>= 0.5;
Etr = tasa_error(ytr_pred,ytr);
disp(Etr);

%tasa de error test
ytest_pred = 1./(1+exp(-(Xtest*theta)))>= 0.5;
Etest = tasa_error(ytest_pred,ytest);
disp(Etest);

%dibujar frontera de decisión
plotDecisionBoundary(theta, Xtr, ytr);
xlabel('Exam 1 score');
ylabel('Exam 2 score');

%% Ejemplo alumno
segExamen = 1:100;
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
lambda = logspace(-6, 2);    %vector de lambdas logarítimico

%elegir mejor parámetro de regularización
[best_lambda,Etr,Ecv] = kfold_cross_validation(Xtr_exp,ytr,10,lambda');

%Dibujar evolución tasas de error de entrenamiento y validación
figure;
grid on; hold on;
ylabel('Tasa de error'); xlabel('Factor de regularización');

plot(log10(lambda), Etr, 'r-');
plot(log10(lambda), Ecv, 'b-');
plot(log10(lambda(best_lambda)), Ecv(best_lambda), 'g.');
legend('Tasa de error de entrenamiento', 'Tasa de error de validación', 'Punto ideal');

%entrenar todos los datos mejor modelo
th1 = minFunc(@costeLogReg,zeros(size(Xtr_exp,2),1), options, Xtr_exp, ytr, lambda(best_lambda));

%entrenar todos los datos lambda = 0
th2 = minFunc(@costeLogReg,zeros(size(Xtr_exp,2),1), options, Xtr_exp, ytr, 0);

%dibujar superficie de separación mejor modelo
plotDecisionBoundary(th1, Xtr_exp, ytr);
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

%dibujar superficie de separación lambda = 0
plotDecisionBoundary(th2, Xtr_exp, ytr);
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

ytr_pred1 = 1./(1+exp(-(Xtr_exp*th1))) >= 0.5;
Etr1 = tasa_error(ytr_pred1,ytr)

ytr_pred2 = 1./(1+exp(-(Xtr_exp*th2))) >= 0.5;
Etr2 = tasa_error(ytr_pred2,ytr)
%% Precisión/Recall

% calcular matriz de confusión
Xtest_exp = mapFeature(Xtest(:,1),Xtest(:,2));    %expansión de funciones base

ytest_pred = 1./(1+exp(-(Xtest_exp*th1))) >= 0.5;
Etest = tasa_error(ytest_pred,ytest)

TP = sum(ytest_pred == 1 & ytest == 1);
FP = sum(ytest_pred == 1 & ytest == 0);
TN = sum(ytest_pred == 0 & ytest == 0);
FN = sum(ytest_pred == 0 & ytest == 1);

P = TP/(TP+FP);
R = TP/(TP+FN);

M_CONF = [TP FP; TN FN];
disp(M_CONF);

% Buscar umbral que consiga un 95% de precisión
for umbral=0:0.01:1
    ytest_pred = 1./(1+exp(-(Xtest_exp*th1))) >= umbral;
    TP = sum(ytest_pred == 1 & ytest == 1);
    FP = sum(ytest_pred == 1 & ytest == 0);
    P_buscada = TP/(TP+FP);
    if(P_buscada >= 0.95)
        disp(umbral);
        break
    end
end