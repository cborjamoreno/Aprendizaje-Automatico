%% Cargar los datos de entrenamiento
data_train = load('PisosTrain.txt');
y_train = data_train(:,3);  % Precio en Euros
x1_train = data_train(:,1); % m^2
x2_train = data_train(:,2); % Habitaciones
N_train = length(y_train);

%% Cargar los datos de test
data_test = load('PisosTest.txt');
y_test = data_test(:,3);  % Precio en Euros
x1_test = data_test(:,1); % m^2
x2_test = data_test(:,2); % Habitaciones
N_test = length(y_test);

%% limpiador
clear
close all
%% 2. Regresión monovariable en función de la superficie usando ecuación normal
figure;
plot(x1_train, y_train, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on; 

X_train = [ones(N_train,1) x1_train];
th = X_train \ y_train;  % Aplicar ecuación normal
fprintf("%2f + %2f*x1\n", th(1,1), th(2,1));
Xextr_train = [1 min(x1_train)  % Predicción para los valores extremos
         1 max(x1_train)];
yextr_train = Xextr_train * th;
plot(Xextr_train(:,2), yextr_train, 'r-'); % Dibujo la recta de predicción
xlim([0 350]);
ylim([0 12*10^5]);
legend('Datos Entrenamiento', 'Prediccion')

% Cálculo del error RMS para los datos de entrenamiento
rmse_train = sqrt(calcularSSE(th,X_train,y_train) / N_train);
fprintf("RMSE Data train: %2f\n", rmse_train);

% Gráfica con los datos de test y la recta de predicción calculada
figure;
plot(x1_test, y_test, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on;
plot(Xextr_train(:,2), yextr_train, 'r-');
xlim([0 350]);
ylim([0 12*10^5]);
legend('Datos Entrenamiento', 'Prediccion')

% Cálculo del error RMS para los datos de test
rmse_test = sqrt(calcularSSE(th,X_test,y_test) / N_test);
fprintf("RMSE Data test: %2f\n", rmse_test);

%% 3. Regresión multivariable en función de la superficie y el número de
%     habitaciones

X_train = [ones(N_train,1) x1_train x2_train];

%Normalilzar
N = size(X_train,1);
mu = mean(X_train(:,2:end));
sig = std(X_train(:,2:end));
X_train(:,2:end) = (X_train(:,2:end) - repmat(mu,N,1))./ repmat(sig,N,1);

th = X_train \ y_train;  % Aplicar ecuación normal
yest_train = X_train * th;

%Des-normalizar
th(2:end) = th(2:end)./sig';
th(1) = th(1)-(mu * th(2:end));
fprintf("%2f + %2f*x1 + %2f*x2\n", th);

% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1_train, x2_train, y_train, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1_train x1_train]' , [x2_train x2_train]' , [y_train yest_train]', '-b');

% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1_train), max(x1_train), np)';
ejex2 = linspace(min(x2_train), max(x2_train), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = Xg * th;

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');

% Cálculo del error RMS para los datos de entrenamiento
rmse_train = sqrt(calcularSSE(th,X_train,y_train) / N_train);
fprintf("RMSE Data train: %2f\n", rmse_train);

% Dibujar los datos de test
X_test = [ones(N_test,1) x1_test x2_test];
yest_test = X_test * th;
% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1_test, x2_test, y_test, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1_test x1_test]' , [x2_test x2_test]' , [y_test yest_test]', '-b');

% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1_test), max(x1_test), np)';
ejex2 = linspace(min(x2_test), max(x2_test), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = Xg * th;

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');

% Cálculo del error RMS para los datos de test
rmse_test = sqrt(calcularSSE(th,X_test,y_test) / N_test);
fprintf("RMSE Data test: %2f\n", rmse_test);
