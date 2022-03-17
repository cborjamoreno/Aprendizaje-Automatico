function [best_model,Etr,Ecv] = kfold_cross_validation(X, y, k, models)
%k-fold para elegir el valor del parámetro de regularizacion
    best_model = 0;
    best_errV = inf;
    options = [];
    options.display = 'final'; %otros: 'iter' , 'none‘
    options.method = 'newton'; %por defecto: 'lbfgs'

    Etr = [];
    Ecv = [];
    [rows,~] = size(models);
    for model = 1:rows
        etr = 0; ecv = 0;
        for fold = 1:k
            [Xcv,ycv,Xtr,ytr] = particion(fold,k,X,y);
            th = minFunc(@costeLogReg,zeros(size(Xtr,2),1), options, Xtr, ytr, models(model));

            h = 1./(1+exp(-(Xtr*th))) >= 0.5;
            etr = etr + tasa_error(h,ytr);

            h = 1./(1+exp(-(Xcv*th))) >= 0.5;
            ecv = ecv + tasa_error(h,ycv);

        end
        etr = etr / k;
        Etr = [Etr;etr];
        ecv = ecv / k;
        Ecv = [Ecv;ecv];
        if ecv < best_errV
            best_errV = ecv;
            best_model = model;
        end
    end
end

