function [best_model,Etr,Ecv] = kfold_cross_validation(X, y, k, models)
%k-fold para elegir el valor del parámetro de regularizacion
    best_model = 0;
    best_errV = inf;
    Etr = [];
    Ecv = [];
    [rows,~] = size(models);
    for model = 1:rows
        etr = 0; ecv = 0;
        for fold = 1:k
            [Xcv,ycv,Xtr,ytr] = particion(fold,k,X,y);
            H = Xtr'*Xtr + models(model)*diag([0 ones(1,size(Xtr,2)-1)]);
            th = H \ (Xtr'*ytr);

            h = 1./(1+exp(-(Xtr*th)));
            ytr_pred = double(h >= 0.5);
            etr = etr + tasa_error(ytr_pred,ytr);

            h = 1./(1+exp(-(Xcv*th)));
            ycv_pred = double(h >= 0.5);
            ecv = ecv + tasa_error(ycv_pred,ycv);

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

