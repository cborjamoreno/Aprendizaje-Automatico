function munew = updateCentroids(D,c,k)
% D((m,n), m datapoints, n dimensions
% c(m) assignment of each datapoint to a class
%
% munew(K,n) new centroids

munew = zeros(k,3);

for i=1:k
    v = D(c==i,:);
    mean(v);
    munew(i,:) = mean(v);
end
