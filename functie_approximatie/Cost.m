%Kostfunctie voor een gegeven set van trainingdata en gegeven biases en
%gewichten
    %INPUT:
    %TD: Trainingdata, in de vorm van een tabel
    %B: cell array van biases
    %W: cell array van gewichten

    %OUTPUT:
    %C: Output van de kostfunctie voor de gegeven biases en gewichten
function C = Cost(TD,B,W)
m = height(TD);
C = 0;
for i = 1:m
    C = C + norm(NN(TD.x(i),B,W)-TD.y(i)).^2;
end
C = C/m;
end