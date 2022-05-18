function [mu, c, J] = kmeans(D,mu0)

% D(m,n), m datapoints, n dimensions
% mu0(K,n) K initial centroids
%
% mu(K,n) final centroids
% c(m) assignment of each datapoint to a class

J = [];

k = size(mu0,1);
c = updateClusters(D,mu0);
mu = updateCentroids(D,c,k);

while 1
    for j=1:size(D,1)
        muc_j = mu(c(j),:);
    end
    j = sum(funcionDistorsion(D,muc_j))/3;
    J = [J j];
    c2 = updateClusters(D,mu);
    if (isequal(c,c2))
        break;
    end
    mu = updateCentroids(D,c2,k);
    c = updateClusters(D,mu);
end


