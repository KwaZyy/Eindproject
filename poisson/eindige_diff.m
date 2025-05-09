m = input("m: ");
h = 1/(m+1);
n = m^2;
f = @(x,y) 2*x*(1-x)+2*y*(1-y);
g = @(x,y) x.^3 - 3.*x.*y^2;
u = @(x,y) x*(1-x)*y*(1-y)+x.^3 - 3.*x.*y^2; %Werkelijke oplossing
f_ij = zeros(n,1);
u_ij = zeros(n,1);
for i = 1:m
    for j = 1:m
        a = i+(j-1)*m; %index
        u_ij(a) = u(i*h,j*h);
        if i == 1 && j ~= 1 && j ~= m
            f_ij(a) = f(i*h,j*h) + g(0,j*h)/h^2;
        elseif i == m && j ~= 1 && j ~= m
            f_ij(a) = f(i*h,j*h) + g(1,j*h)/h^2;
        elseif j == 1 && i ~= 1 && i ~= m
            f_ij(a) = f(i*h,j*h) + g(i*h,0)/h^2;
        elseif j == m && i ~= 1 && i ~= m
            f_ij(a) = f(i*h,j*h) + g(i*h,1)/h^2;
        else
            f_ij(a) = f(i*h,j*h);
        end
    end
end
%randpunten
f_ij(1) = f(h,h) + g(0,h)/h^2 + g(h,0)/h^2; %(1,1)
f_ij(m) = f(m*h,h) + g(1,h)/h^2 + g(m*h,0)/h^2; %(m,1)
f_ij((m-1)*m+1) = f(h,m*h) +g(0,m*h)/h^2 + g(h,1)/h^2;  %(1,m)
f_ij(n) = f(m*h,m*h) + g(1,m*h)/h^2 + g(m*h,1)/h^2; %(m,m)


%Discretisatiematrix
A = zeros(n,n);

%4Hoekpunten
A(1,1) = 4/h.^2; A(1,2) = -1/h.^2; A(1,m+1) = -1/h.^2;   %(1,1)
A(m,m) = 4/h.^2; A(m,m-1) = -1/h.^2; A(m,m+m) = -1/h.^2; %(m,1)
A((m-1)*m+1,(m-1)*m+1) = 4/h.^2; A((m-1)*m+1,(m-1)*m+1+1) = -1/h.^2; A((m-1)*m+1,(m-1)*m+1-m) = -1/h.^2; %(1,m)
A(m^2,m^2) = 4/h.^2; A(m^2,m^2-1) = -1/h.^2; A(m^2,m^2-m) = -1/h.^2; %(m,m)
%randvoorwaarden
for k = 2:m-1
    %onderzijde
    A(k,k) = 4/h.^2;
    A(k,k-1) = -1/h.^2; %links
    A(k,k+1) = -1/h.^2; %rechts
    A(k,k+m) = -1/h.^2; %boven
    %bovenzijde
    A((m-1)*m+k,(m-1)*m+k) = 4/h.^2;
    A((m-1)*m+k,(m-1)*m+k-1) = -1/h.^2; %links
    A((m-1)*m+k,(m-1)*m+k+1) = -1/h.^2; %rechts
    A((m-1)*m+k,(m-1)*m+k-m) = -1/h.^2; %onder
    %linkerzijde
    A((k-1)*m+1,(k-1)*m+1) = 4/h.^2;
    A((k-1)*m+1,(k-1)*m+1+1) = -1/h.^2; %rechts
    A((k-1)*m+1,(k-1)*m+1-m) = -1/h.^2; %onder
    A((k-1)*m+1,(k-1)*m+1+m) = -1/h.^2; %boven
    %rechterzijde
    A(k*m,k*m) = 4/h.^2;
    A(k*m,k*m-1) = -1/h.^2; %linker
    A(k*m,k*m-m) = -1/h.^2; %onder
    A(k*m,k*m+m) = -1/h.^2; %boven
end
for i = 2:m-1
    for j = 2:m-1
        a = i+(j-1)*m; %index
        A(a,a) = 4/h.^2;
        A(a,a-1) = -1/h.^2; %links
        A(a,a+1) = -1/h.^2; %rechts
        A(a,a-m) = -1/h.^2; %onder
        A(a,a+m) = -1/h.^2; %boven      
    end
end
v = A\f_ij;
V = zeros(m+2,m+2);
for i = 1:m
    for j = 1:m
        a = i+(j-1)*m;
        V(i+1,j+1) = abs(v(a)-u_ij(a));
    end
end
V = V';
%Contourplot
hold off
x = linspace(0,1,m+2);
y = linspace(0,1,m+2);
[X, Y] = meshgrid(x, y);
contourf(X,Y,V)
colorbar;
hold on
title("Contourplot van de fout $|u(x,y)-v_{i,j}|$", 'Interpreter','latex')
xlabel("x")
ylabel("y")