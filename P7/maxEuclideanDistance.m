function [max_p] = maxEuclideanDistance(mat_d)
    dmax = 0;
    ind_max = 0;
    for i=1:size(mat_d,1)
        d = sum(mat_d(i,:));
        if d > dmax
            dmax = d;
            ind_max = i;
        end
    end
    max_p = ind_max;
end

