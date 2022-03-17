function [E] = tasaError(salidas,y)
%Calcula la tasa de error
    E = sum(salidas ~= y)/length(y);
end

