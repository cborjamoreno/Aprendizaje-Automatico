function [theta,RMSEtr,RMSEcv] = kfold_cross_validation(reg, X, y, k, models)
    best_model = 0;
    best_errV = inf;
    RMSEtr = [];
    RMSEcv = [];
    [rows,~] = size(models);
    for model = 1:rows
        err_T = 0; err_V = 0;
        for fold = 1:k
            [Xcv,ycv,Xtr,ytr] = particion(fold,k,X,y);
            if(reg) %regularizacion
                H = Xtr'*Xtr + models(model)*diag([0 ones(1,length(models))]);
                th = H \ (Xtr'*ytr);
            else
                Xtr_exp = expandir(Xtr,models(model,:));
                Xcv_exp = expandir(Xcv,models(model,:));
                [Xn,mu,sig] = normalizar(Xtr_exp);
                th = Xn \ ytr;
                [th] = desnormalizar(th,mu,sig);
            end
            err_T = err_T + RMSE(th,Xtr_exp,ytr);
            err_V = err_V + RMSE(th,Xcv_exp,ycv);
        end
        err_T = err_T / k;
        RMSEtr = [RMSEtr;err_T];
        err_V = err_V / k;
        RMSEcv = [RMSEcv;err_V];
        if err_V < best_errV
            best_model = model;
        end
    end
    if(reg) %regularizacion
         H = Xtr'*Xtr + models(best_model)*diag([0 ones(1,length(models))]);
         theta = H \ (Xtr'*ytr);
    else
        Xtr_exp = expandir(Xtr,models(best_model,:));
        Xcv_exp = expandir(Xcv,models(best_model,:));
        theta = Xn \ ytr;
end

