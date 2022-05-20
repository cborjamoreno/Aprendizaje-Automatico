% clear all;
% close all;
%% imagen
figure(1)
im = imread('smallparrot.jpg');
% im = imread('atardecer.jpeg');
% im = imread('elcolor.jpg');
imshow(im)

%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));

%% dimensiones
m = size(D,1);
n = size(D,2);

%% Kmeans 
array_k = [2 4 8 16 32];
%% Escoger agrupamiento
J_k = [];

for i=1:size(array_k,2)


    % inicialización de los centroides en muestras aleatorias
%     rng('shuffle');
%     ind_clusters = fix((m).*rand(K,1))';
%     mu0 = D(ind_clusters,:);


    K = array_k(i);
    mu0 = initCentroids(D,K);

    %bucle kmeans
    [mu, c, J] = kmeans(D, mu0);
    j = min(J);
    J_k = [J_k j];

end

figure(3);
plot(array_k, J_k, 'r-', 'LineWidth',1);
title('Evolución coste')
ylabel('Coste'); xlabel('K');



%% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K
    ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure(2)
imshow(uint8(qIM));