function [m,P,R] = confusionMatrix(ytest_pred, ytest, N_clases)
% Calcular matriz de confusión y precisión/recall

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
P(N_clases) = m(N_clases,N_clases)/sum(m(N_clases,:));

% Cálculo recall de cada clase
for i=1:N_clases-1
    R(i) = m(i,i)/sum(m(:,i));
end
R(N_clases) = m(N_clases,N_clases)/sum(m(:,N_clases));
end

