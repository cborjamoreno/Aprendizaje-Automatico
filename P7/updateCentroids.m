function munew = updateCentroids(D,c)
% D((m,n), m datapoints, n dimensions
% c(m) assignment of each datapoint to a class
%
% munew(K,n) new centroids

k = height(unique(c));
munew = zeros(k,3);

for i=1:k
    v = D(c==i,:);
    mean(v);
    munew(i,:) = mean(v);
end
