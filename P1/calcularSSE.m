function [sse] = calcularSSE(theta,X,y)
    r = X*theta - y;
    sse = r'*r;
end