function J = funcionDistorsion(x,mu)
    r = x-mu;
    J = sum(r.^2)/size(x,2);
end
