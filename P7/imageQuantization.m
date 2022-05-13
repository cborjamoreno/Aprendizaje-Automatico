figure(1)
im = imread('smallparrot.jpg');
% imshow(im)

%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));

%% dimensiones
m = size(D,1);
n = size(D,2);

%% Kmeans 
K = 16;

%% Escoger agrupamiento

for i=1:5
    % inicialización aleatoria
    mu0 = zeros(K,3);
    rng('shuffle');
    for i=1:K
        mu0(i,:) = fix((255).*rand(3,1))';
    end
    
    %bucle kmeans
    [mu, c] = kmeans(D, mu0);

    %calculo J
    J = funcionDistorsion(D,mu);
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