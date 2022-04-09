function modelo = entrenarGaussianas( Xtr, ytr, nc, NaiveBayes, landa )
% Entrena una Gaussana para cada clase y devuelve:
% modelo{i}.N     : Numero de muestras de la clase i
% modelo{i}.mu    : Media de la clase i
% modelo{i}.Sigma : Covarianza de la clase i
% Si NaiveBayes = 1, las matrices de Covarianza serán diagonales
% Se regularizarán las covarianzas mediante: Sigma = Sigma + landa*eye(D)

[~,D] = size(Xtr);

for i=1:nc
    [Ni,~] = size(find(ytr==i));
    Xi = Xtr((ytr==i),:);
    modelo{i}.N = Ni; 
    modelo{i}.mu = mean(Xi);
    modelo{i}.Sigma = cov(Xi);

    if NaiveBayes == 1
        modelo{i}.Sigma = diag(diag(modelo{i}.Sigma));
    end

    modelo{i}.Sigma = modelo{i}.Sigma + landa*eye(D);

end

