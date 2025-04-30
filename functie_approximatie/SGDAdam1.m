%Adam Stochastische Gradient Descent, t neemt toe na elke iteratie
tic
m = height(TD);
rng(2000)
beta1 = 0.9;
beta2 = 0.999;
epsilon = 10^(-8);
mtweight = cellfun(@minus,W,W,'Un',0);
mthatweight = cellfun(@minus,W,W,'Un',0);
mtbias = cellfun(@minus,B,B,'Un',0);
mthatbias = cellfun(@minus,B,B,'Un',0);
vtweight = cellfun(@minus,W,W,'Un',0);
vthatweight = cellfun(@minus,W,W,'Un',0);
vtbias = cellfun(@minus,B,B,'Un',0);
vthatbias = cellfun(@minus,B,B,'Un',0);
costlist3 = zeros(1,N);
index = 1:m;
for t = 0:N-1
    old_cost = Cost(TD,B,W);
    for i = 1:m
        [grad_B,grad_W] = Backpropagation(TD.x(index(i)),TD.y(index(i)),B,W);
        grad_B = cellfun(@(x) x * n, grad_B, 'UniformOutput', false);
        grad_W = cellfun(@(x) x * n, grad_W, 'UniformOutput', false);

        mtweight = cellfun(@(x,y) beta1.*x + (1-beta1).*y,mtweight,grad_W,'UniformOutput',false);
        mtbias = cellfun(@(x,y) beta1.*x + (1-beta1).*y,mtbias,grad_B,'UniformOutput',false);
        vtweight = cellfun(@(x,y) beta2.*x + (1-beta2).*(y.^2),vtweight,grad_W,'UniformOutput',false);
        vtbias = cellfun(@(x,y) beta2.*x + (1-beta2).*(y.^2),vtbias,grad_B,'UniformOutput',false);
        %mthat
        mthatweight = cellfun(@(x) x./(1-beta1^(t+1)), mtweight, 'UniformOutput', false);
        mthatbias = cellfun(@(x) x./(1-beta1^(t+1)), mtbias, 'UniformOutput', false);
        %vthat
        vthatweight = cellfun(@(x) x./(1-beta2^(t+1)), vtweight, 'UniformOutput', false);
        vthatbias = cellfun(@(x) x./(1-beta2^(t+1)), vtbias, 'UniformOutput', false);

        B0 = cellfun(@(x,y,z) x -n.*y./(sqrt(z)+epsilon),B,mthatbias,vthatbias,'UniformOutput',false);
        W0 = cellfun(@(x,y,z) x -n.*y./(sqrt(z)+epsilon),W,mthatweight,vthatweight,'UniformOutput',false);
    end
    new_cost = Cost(TD,B0,W0)
    costlist3(t+1) = new_cost;
    index = index(randperm(length(index)));
    B = B0;
    W = W0;
end