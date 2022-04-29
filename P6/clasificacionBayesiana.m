function yhat = clasificacionBayesiana(modelo, X)
% Con los modelos entrenados, predice la clase para cada muestra X
    ypred = [];
    for i = 1:width(modelo)
        Pi = modelo{i}.N / height(X);
        % se multiplica por la probabilidad a priori de la clase i
        ypred(:, i) = gaussLog(modelo{i}.mu,modelo{i}.Sigma,X) * Pi;
    end
    [~,yhat] = max(ypred,[],2);
end




