function [Xtr, ytr, Xtest, ytest] = separar(X, y, train_percent)
%Separa data en datos de entrenamiento y datos de test. Utilizando
%train_percent como porcentaje de datos de entrenamiento.

    N = size(X,1);
    idx = randperm(N);
    Xtr = X(idx(1:round(N*train_percent)),:);
    ytr = y(idx(1:round(N*train_percent)),:);
    Xtest = X(idx(round(N*train_percent)+1:end),:);
    ytest = y(idx(round(N*train_percent)+1:end),:);

end

