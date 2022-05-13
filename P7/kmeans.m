function [mu, c] = kmeans(D,mu0)

% D(m,n), m datapoints, n dimensions
% mu0(K,n) K initial centroids
%
% mu(K,n) final centroids
% c(m) assignment of each datapoint to a class

c = updateClusters(D,mu0);
mu = updateCentroids(D,c);
cnew = updateClusters(D,mu);
while(c ~= cnew)
    c = updateClusters(D,mu);
    mu = updateCentroids(D,c);
    cnew = updateClusters(D,mu);
end


