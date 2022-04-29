function [E] = tasa_error(salidas,y)
%Calcula la tasa de error
    E = sum(salidas ~= y)/height(y);
end

