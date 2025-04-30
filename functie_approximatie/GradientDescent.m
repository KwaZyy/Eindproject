f = @(x,y) (x-pi).^2+(y-exp(1)).^2;
grad_f = @(x,y) [2*(x-pi),2*(y-exp(1))];
n = 0.1;
eps = 0.01;

x_0 = [9,10];
x_1 = x_0 - n*grad_f(x_0(1),x_0(2));
k = 0;

x_k = x_0;
x_k1 = x_1;
X = 0;
while norm(x_k1 - x_k) >= eps
    x_k = x_k1;
    x_k1 = x_k - n*grad_f(x_k(1),x_k(2))
    k = k+1;
    X(k,1) = x_k1(1)
    X(k,2) = x_k1(2)
end
grid on
plot(X(:,1),X(:,2))