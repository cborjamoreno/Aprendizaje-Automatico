function mu0 = initCentroids(D,K)
    % matriz que los índices de las muestras donde inicializar los
    % clusters
    ind_clusters = [];
    m = size(D,1);
    D = sort(D);

    rng('shuffle');
    nMuestrasCluster = m/K;
    for i=1:K
        min = (i-1)*nMuestrasCluster + 1;
        max = i*nMuestrasCluster;
        ind_clusters = [ind_clusters randi([min max],1)];
    end
    mu0 = D(ind_clusters,:);
end

