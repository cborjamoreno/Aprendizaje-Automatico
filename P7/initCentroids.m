function mu0 = initCentroids(D,K)
% Inicialización del algoritmo k-means utilizando la heurística "furthest"
    m = size(D,1);

    % matriz que almacena los índices de las muestras en las que están
    % situados los clusters
    ind_clusters = [];

    % 1er cluster es una muestra aleatoria
    rng('shuffle');
    ind_clusters = [ind_clusters randi([1 m],1)];

    % 2º cluster es la muestra más alejada del 1er cluster.
    [~, dist] = dsearchn(D(ind_clusters,:),D);
    [~,new_cluster] = max(dist);
    ind_clusters = [ind_clusters new_cluster];

    for i=2:K-1
        % Se calcula la distancia euclídea desde cada muestra a su cluster más
        % cercano.
        [~, dist] = dsearchn(D(ind_clusters,:),D);

        % El cluster i es la muestra que tiene la distancia máxima con
        % respecto a su cluster más cercano
        [~,new_cluster] = max(dist);
        ind_clusters = [ind_clusters new_cluster];
    end
    mu0 = D(ind_clusters,:);
end

