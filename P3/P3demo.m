%% Based on exercise 2 of Machine Learning Online Class by Andrew Ng 

clear ; close all;

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('exam_data.txt');
[dataTrain, dataTest]separar(data,0.2);
ytrain = dataTrain(:, 3);
Xtrain = dataTrain(:, [1, 2]);
N = length(ytrain);
ytest = dataTest(:, 3);
Xtest = dataTest(:, [1, 2]);



plotData(X, y);
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')


%% Calcula una Solucion Absurda
X = [ones(N,1) X];
theta = [-70 0.7 0.3]'; 

%% Dibujar la Solucion 
plotDecisionBoundary(theta, X, y);
xlabel('Exam 1 score');
ylabel('Exam 2 score');


