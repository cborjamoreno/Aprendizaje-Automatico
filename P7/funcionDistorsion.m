function J = funcionDistorsion(x,mu,c)
    m = size(x,1);
    J = sum(sum((x-mu(c,:)).^2,2))./m;
end


