function [grad_B,grad_W] = Backpropagation(x,y,B,W)
[~,A,Z] = NN(x,B,W);
L = length(B); %Aantal lagen
grad_A = cellfun(@minus,B,B,'Un',0);
grad_W = cellfun(@minus,W,W,'Un',0);
grad_B = grad_A;
for i = 0:(L-2) %Per laag (achterstevoren)
    for j = 1:length(B{L-i}) %Aantal neuronen/biases per laag
        if i == 0 %Laatste laag geen activatiefunctie
            grad_A{L-i}(j) = 2*(A{L-i}(j)-y(j));
            grad_B{L-i}(j) = grad_A{L-i}(j);
            for k = 1:length(B{L-i-1})
                grad_W{L-i}(j,k) = grad_B{L-i}(j) * A{L-i-1}(k);
            end
        else
            for m = 1:length(B{L-i+1})
                %da/dz en dz/da  berekenen
                dadz = DiffReLU(Z{L-i+1}(m));
                dzda = W{L-i+1}(m,j);
                grad_A{L-i}(j) = grad_A{L-i}(j) + grad_A{L-i+1}(m)*dadz*dzda;
            end
            grad_B{L-i}(j) = grad_A{L-i}(j)*DiffReLU(Z{L-i}(j));
            for k = 1:length(B{L-i-1})
            grad_W{L-i}(j,k) = grad_B{L-i}(j) * A{L-i-1}(k);
            end
        end
    end
end