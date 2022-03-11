function [best_model,RMSEtr,RMSEcv] = kfold_cross_validation(X, y, k, models)
%k-fold para elegir el valor del parámetro de regularizacion
    best_model = 0;
    best_errV = inf;
    RMSEtr = [];
    RMSEcv = [];
    [rows,~] = size(models);
    for model = 1:rows
        err_T = 0; err_V = 0;
        for fold = 1:k
            [Xcv,ycv,Xtr,ytr] = particion(fold,k,X,y);

            [Xn,mu,sig] = normalizar(Xtr);
            H = Xn'*Xn + models(model)*diag([0 ones(1,size(Xn,2)-1)]);
            th = H \ (Xn'*ytr);
            [th] = desnormalizar(th,mu,sig);

            err_T = err_T + RMSE(th,Xtr,ytr);
            err_V = err_V + RMSE(th,Xcv,ycv);
        end
        err_T = err_T / k;
        RMSEtr = [RMSEtr;err_T];
        err_V = err_V / k;
        RMSEcv = [RMSEcv;err_V];
        if err_V < best_errV
            best_errV = err_V;
            best_model = model;
        end
    end
end

