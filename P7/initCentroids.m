function mu0 = initCentroids(D,K)
    m = size(D,1);

    % matriz que almacena las distancias euclideas desde cada cluster a las
    % demás muestras
    mat_d = [];
    ind_clusters = zeros(K,1);
    rng(0,'twister');
    ind_clusters(1,:) = randi([1 m],1);

    for i=2:K
        v_d = euclideanDistance(ind_clusters(i-1,:),D)';
        mat_d = [mat_d v_d];
        ind_clusters(i,:) = maxEuclideanDistance(mat_d);
    end
    mu0 = D(ind_clusters,:);
end

