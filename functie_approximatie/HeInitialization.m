%Deze functie genereert willekeurige biases en gewichten voor een neuraal
%netwerk
    %INPUT:
    %l_2: Vector met het aantal neuronen voor elke tussenlaag in het
    %netwerk:

    %OUTPUT:
    %B: cell array van biases
    %W: cell array van gewichten

function [B,W] = HeInitialization(l_2)
L = length(l_2)+2;
l = ones(1,L); l(2:length(l_2)+1) = l_2;
B = {0};
W = {0};
for j = 2:L
    b_temp = zeros(l(j),1);
    w_temp = normrnd(0,sqrt(2/l(j)),[l(j),l(j-1)]);
    B{end+1} = b_temp;
    W{end+1} = w_temp;
end
end