function Z = updateClusters(D,mu)
% D(m,n), m datapoints, n dimensions
% mu(K,n) final centroids
%
% c(m) assignment of each datapoint to a class

Z = dsearchn(mu,D);