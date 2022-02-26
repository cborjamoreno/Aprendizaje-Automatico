function[X_norm] = normalizar(X,N)
    X_norm = X;
    N = size(X_norm,1);
    mu = mean(X_norm(:,2:end));
    sig = std(X_norm(:,2:end));
    X_norm(:,2:end) = (X_norm(:,2:end) - repmat(mu,N,1))./ repmat(sig,N,1);
end