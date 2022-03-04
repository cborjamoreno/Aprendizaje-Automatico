close all;
clear;
%% Cargar los datos
datos = load('CochesTrain.txt');
ydatos = datos(:, 1);   % Precio en Euros
Xdatos = datos(:, 2:4); % Años, Km, CV

datos2 = load('CochesTest.txt');
ytest = datos2(:,1);  % Precio en Euros
Xtest = datos2(:,2:4); % Años, Km, CV
Ntest = length(ytest);

%% Selección del grado del polinomio para la antigüedad del coche
models=[1 1 1 ; 2 1 1; 3 1 1; 4 1 1; 5 1 1; 6 1 1; 7 1 1; 8 1 1; 9 1 1; 10 1 1];
[w,RMSEtr,RMSEcv] = kfold_cross_validation(false,Xdatos,ydatos,10,models);

figure;
grid on; hold on;
ylabel('Coste'); xlabel('Complejidad');

plot((1:length(RMSEtr)), RMSEtr, 'r-'); % Dibujo la recta de predicción
plot((1:length(RMSEcv)), RMSEcv, 'b-'); % Dibujo la recta de predicción
legend('Error de entrenamiento')


%% Dibujo de un Ajuste Parabólico Monovariable
