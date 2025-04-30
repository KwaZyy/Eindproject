%Stochastische gradient descent
tic
rng(2000) %Voor reproduceerbaarheid
m = height(TD);
index = 1:m;
B_hold = {};
W_hold = {};
costlist3 = zeros(1,N);
for j = 1:N
    old_cost = Cost(TD,B,W);
    for i = 1:m
        [grad_B,grad_W] = Backpropagation(TD.x(index(i)),TD.y(index(i)),B,W);
        temp1 = cellfun(@(x) x * n, grad_B, 'UniformOutput', false);
        temp2 = cellfun(@(x) x * n, grad_W, 'UniformOutput', false);
        B0 = cellfun(@minus,B,temp1,'Un',0);
        W0 = cellfun(@minus,W,temp2,'Un',0);
    end
    new_cost = Cost(TD,B0,W0)
    costlist3(j) = new_cost;
    index = index(randperm(length(index)));
    if new_cost < old_cost %Dit werkt niet, we willen ze allemaal vergelijken
        %Met elkaar
        B_hold = B0;
        W_hold = W0;
        new_cost = Cost(TD,B0,W0)
    end
    B = B0;
    W = W0;
end
time3 = toc
B = B_hold;
W = W_hold;
TD.NNx = transpose(NN(TD.x,B,W));
hold off
semilogy(1:N,costlist,'LineWidth',2)
hold on
semilogy(1:N,costlist3,'LineWidth',1)
semilogy(1:N,costlist2,'LineWidth',2)
grid on
title("Kostfuncties van verschillende gradient descent methoden")
xlabel("Iteratie van gekozen methode")
ylabel("Kost")
legend('Gradient descent','Stochastische gradient descent','Adam')