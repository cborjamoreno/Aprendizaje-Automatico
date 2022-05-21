% clear all;
% close all;
%% imagen
figure(1)
% Descomentar la imagen que se quiera probar
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
% Si no se quiere sacar la evolución de J, se recomienda elegir un valor de
% K fijo por cada ejecución (e.g. array_k = [2]) para disminuir el tiempo
% de ejecución

print_evolucion = true; % false si no se quiere mostrar la evolución de J y T
array_k = [2 4 8 16 32 64]; % disminuir a un solo valor de K si no se quiere mostrar la evolución de J
T = []; % almacén del tiempo de ejecución para cada K
%% Escoger agrupamiento
J_k = [];

for i=1:size(array_k,2)

    tic
    % inicialización de los centroides en muestras aleatorias
    rng('shuffle');
    ind_clusters = fix((m).*rand(K,1))';
    mu0 = D(ind_clusters,:);


    K = array_k(i);
%     mu0 = initCentroids(D,K);

    %bucle kmeans
    [mu, c, J] = kmeans(D, mu0);
    j = min(J);
    J_k = [J_k j];

    t = toc;
    T = [T t];
    
    ahorro_percent = ((m*n-(m+K*n))/(m*n))*100;
    fprintf("Ahorro de espacio con K = %d: %.2f\n", K,ahorro_percent);
    fprintf("Error de reconstrucción con K = %d: %.2f\n",K,J_k(i));

end

if(print_evolucion)
    figure(3);
    plot(array_k, J_k, 'r-', 'LineWidth',1);
    ylabel('J'); xlabel('K');

    figure(4);
    plot(array_k, T, 'r-', 'LineWidth',1);
    ylabel('T'); xlabel('K');
end



%% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K
    ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure(2)
imshow(uint8(qIM));