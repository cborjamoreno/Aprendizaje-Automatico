function [J,grad] = costeL2(theta,X,y)
%Calcula el coste cuadrático y el gradiente
    r = X*theta - y;
    J = (1/2)*sum(r.^2);
    grad = X'*r;
end

