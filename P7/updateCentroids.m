function munew = updateCentroids(D,c,k)
% D((m,n), m datapoints, n dimensions
% c(m) assignment of each datapoint to a class
%
% munew(K,n) new centroids

munew = [];

for i=1:k
    v = D(c==i,:);
    media = mean(v);
    munew = [munew;media];
end
