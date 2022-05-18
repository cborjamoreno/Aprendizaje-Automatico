function [v_d] = euclideanDistance2(ind_p,D)
%Calcula la distancia euclidea desde el centroide ind_p hasta cada una de
%las muestras
    v_d = [];
    for i=1:size(D,1)
        p = D(ind_p,:);
        q = D(i,:);
        if(ind_p == i)
            d = 0;
        else 
            d = sqrt((p(:,1)-q(:,1))^2 + (p(:,2)-q(:,2))^2 + (p(:,3)-q(:,3))^2);
        end
        v_d = [v_d d];
    end
end

