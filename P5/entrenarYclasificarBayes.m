function [Etr,Ecv,best_lambda] = entrenarYclasificarBayes(Xtr,ytr,Xcv,ycv, ...
    N_clases,lambda,NaiveBayes)
    
    Etr = [];
    Ecv = [];
    best_lambda = 0;
    best_errV = inf;

    for l = 1:length(lambda)
        etr = 0; ecv = 0;
    
        modelo = entrenarGaussianas(Xtr,ytr,N_clases,NaiveBayes,lambda(l));
        ytr_pred = clasificacionBayesiana(modelo,Xtr);
        ycv_pred = clasificacionBayesiana(modelo,Xcv);

        etr = tasa_error(ytr_pred,ytr);
        ecv = tasa_error(ycv_pred,ycv);

        Etr = [Etr;etr];
        Ecv = [Ecv;ecv];

        if ecv < best_errV
            best_errV = ecv;
            best_lambda = l;
        end
    end
end

