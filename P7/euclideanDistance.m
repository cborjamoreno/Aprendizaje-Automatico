function [m_d] = euclideanDistance(ind_clusters,D)
%Calcula la distancia euclidea desde cada muestra a los centroides
    m_d = [];
    v_d = [];
    for i=1:size(D,1)
        for j=1:size(ind_clusters,2);
            m = D(i,:);
            c = D(j,:);
            d = sqrt((m(:,1)-c(:,1))^2 + (m(:,2)-c(:,2))^2 + (m(:,3)-c(:,3))^2);
            v_d = [v_d d];
        end
        m_d = [m_d v_d];
    end
end

