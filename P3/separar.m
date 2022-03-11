function [dataTrain,dataTest] = separar(data, test_percent)
%Separa data en datos de entrenamiento y datos de test. Utilizando
%test_percent como porcentaje de datos de test.

    cv = cvpartition(size(data,1),'HoldOut',test_percent);
    idx = cv.test;
    
% Separar en datos de entrenamiento y test
    dataTrain = data(~idx,:);
    dataTest  = data(idx,:);
end

