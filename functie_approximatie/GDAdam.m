%Adam Gradient Descent
tic
m = height(TD);
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
costlist2 = zeros(1,N);
for t = 0:N-1
    old_cost = Cost(TD,B,W);
    sum_grad_B = cellfun(@minus,B,B,'Un',0);
    sum_grad_W = cellfun(@minus,W,W,'Un',0);
    for i = 1:m
        [grad_B,grad_W] = Backpropagation(TD.x(i),TD.y(i),B,W);
        sum_grad_B = cellfun(@plus,sum_grad_B,grad_B,'Un',0);
        sum_grad_W = cellfun(@plus,sum_grad_W,grad_W,'Un',0);
    end
    %Dit is de gradiënt voor de kost
    sum_grad_B = cellfun(@(x) x * n/m, sum_grad_B, 'UniformOutput', false);
    sum_grad_W = cellfun(@(x) x * n/m, sum_grad_W, 'UniformOutput', false);

    mtweight = cellfun(@(x,y) beta1.*x + (1-beta1).*y,mtweight,sum_grad_W,'UniformOutput',false);
    mtbias = cellfun(@(x,y) beta1.*x + (1-beta1).*y,mtbias,sum_grad_B,'UniformOutput',false);
    vtweight = cellfun(@(x,y) beta2.*x + (1-beta2).*(y.^2),vtweight,sum_grad_W,'UniformOutput',false);
    vtbias = cellfun(@(x,y) beta2.*x + (1-beta2).*(y.^2),vtbias,sum_grad_B,'UniformOutput',false);
    %mthat
    mthatweight = cellfun(@(x) x./(1-beta1^(t+1)), mtweight, 'UniformOutput', false);
    mthatbias = cellfun(@(x) x./(1-beta1^(t+1)), mtbias, 'UniformOutput', false);
    %vthat
    vthatweight = cellfun(@(x) x./(1-beta2^(t+1)), vtweight, 'UniformOutput', false);
    vthatbias = cellfun(@(x) x./(1-beta2^(t+1)), vtbias, 'UniformOutput', false);

    B0 = cellfun(@(x,y,z) x -n.*y./(sqrt(z)+epsilon),B,mthatbias,vthatbias,'UniformOutput',false);
    W0 = cellfun(@(x,y,z) x -n.*y./(sqrt(z)+epsilon),W,mthatweight,vthatweight,'UniformOutput',false);
    new_cost = Cost(TD,B0,W0)
    costlist2(t+1) = new_cost;
    B = B0;
    W = W0;
end
time2 = toc
TD.NNx = transpose(NN(TD.x,B,W));

%Plot van neural netwerk tegenover oorspronkelijke functie

z = 0:0.01:5;
hold off
plot(z,f(z),'g','LineWidth',2)
hold on
plot(TD.x,TD.y,'go')
plot(z,NN(z,B,W),'b','LineWidth',2)
plot(TD.x,TD.NNx,'bo')
grid on
xlabel("x")
ylabel("y")
legend("$xe^{\sin x}+1$",'Trainingdata $(x_i,y_i)$','$NN(x)$','$(x_i,NN(x_i))$','Interpreter','latex')



%Plot voor het vergelijken van verschillende gradient descent methoden

% hold off
% semilogy(1:N,costlist,'LineWidth',2)
% hold on
% semilogy(1:N,costlist3,'LineWidth',1)
% semilogy(1:N,costlist2,'LineWidth',2)
% grid on
% title("Kostfuncties van verschillende gradient descent methoden")
% xlabel("Iteratie van gekozen methode")
% ylabel("Kost")
% legend('Gradient descent','Stochastische gradient descent','Adam')

%Plot voor relatieve fout trainingdata

% hold off
% plot(z,abs(NN(z,B,W) - f(z))./f(z),'b','LineWidth',2)
% hold on
% plot(TD.x,abs(TD.NNx - TD.y)./TD.y,'bo')
% grid on
% xlabel("x")
% ylabel("y")
% title("Relatieve fout van het neuraal netwerk $NN(x)$ tegenover $f(x) := xe^{\sin x}+1$",'Interpreter','latex')
% legend("Relatieve fout: $|\frac{NN(x) - f(x)}{f(x)}|$","Relatieve fout van trainingdata","Interpreter","latex")