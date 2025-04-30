%Gradient Descent
tic
m = height(TD);
costlist = zeros(1,N);
for j = 1:N
    old_cost = Cost(TD,B,W);
    sum_grad_B = cellfun(@minus,B,B,'Un',0);
    sum_grad_W = cellfun(@minus,W,W,'Un',0);
    for i = 1:m
        [grad_B,grad_W] = Backpropagation(TD.x(i),TD.y(i),B,W);
        sum_grad_B = cellfun(@plus,sum_grad_B,grad_B,'Un',0);
        sum_grad_W = cellfun(@plus,sum_grad_W,grad_W,'Un',0);
    end
    sum_grad_B = cellfun(@(x) x * n/m, sum_grad_B, 'UniformOutput', false);
    sum_grad_W = cellfun(@(x) x * n/m, sum_grad_W, 'UniformOutput', false);
    B0 = cellfun(@minus,B,sum_grad_B,'Un',0);
    W0 = cellfun(@minus,W,sum_grad_W,'Un',0);
    new_cost = Cost(TD,B0,W0)
    costlist(j) = new_cost;
    if new_cost > old_cost
        disp("Minimale kost is gevonden")
        break
    end
    B = B0;
    W = W0;
end
time1 = toc
semilogy(1:N,costlist)
grid on
TD.NNx = transpose(NN(TD.x,B,W));
hold off
%hold off
%plot(TD.x,abs(TD.NNx - TD.y)/TD.y)
%plot(z,abs(NN(z,B,W) - f(z))./f(z))