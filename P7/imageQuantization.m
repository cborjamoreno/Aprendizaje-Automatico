% clear all;
% close all;
%% imagen
figure(1)
% im = imread('smallparrot.jpg');
% im = imread('atardecer.jpeg');
im = imread('elcolor.jpg');
imshow(im)

%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));

%% dimensiones
m = size(D,1);
n = size(D,2);

%% Kmeans 
K = 64;

%% Escoger agrupamiento
iter = 5;
MU = zeros(K,3);
C = zeros(m,iter);
% J = zeros(iter,3);

% for i=1:iter
    % inicialización de los centroides en muestras aleatorias
%     rng('shuffle');
%     ind_clusters = fix((m).*rand(K,1))';
%     mu0 = D(ind_clusters,:);
    mu0 = initCentroids(D,K);

    %bucle kmeans
    [mu, c, J] = kmeans(D, mu0);
%     MU(:,:,i) = mu;
%     C(:,i) = c;

%     for j=1:m
%         muc_j = mu(c(j),:);
%     end

    %calculo J
%     j = funcionDistorsion(D,muc_j);
%     J(i,:) = j;
 
% end

% [j,jindex] = min(sum(J,2));
% mu = MU(:,:,jindex);
% c = C(:,jindex);

disp(J);

% figure(3);
% plot(1:size(J,2), J, 'r-', 'LineWidth',1);
% title('Evolución coste')
% ylabel('Coste'); xlabel('Iteraciones');

%% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K
    ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure(2)
imshow(uint8(qIM));