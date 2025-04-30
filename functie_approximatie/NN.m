%Dit is het bestand waarin de output van een bepaalde invoer berekend wordt
   %INPUT:
   %x: invoeractivatie
   %B: Lijst van alle biases in de vorm van  een cell array voor vectoren.
   %Dit geeft ook het aantal lagen en neuronen per laag aan.
   %W: matrices van alle gewichten in de vorm van een cell array

   %OUTPUT:
   %NNx: De uitvoeractivatie
   %A: De waarde van elke activatie in het neurale netwerk in de vorm van
   %een cell array
   %Z: Geeft de z-waarden aan die gedefinieerd worden bij het onderdeel van
   %backpropagation
function [NNx,A,Z] = NN(x,B,W)
L = length(B);

sigma = @(x) ReLU(x);
NNx = [];
for j = 1:length(x)
    A = {x(j)};
    Z = {x(j)};
    for k = 2:L
        Z{k} = W{k} * A{k-1} + B{k};
        if k == L
            A{k} = Z{k};
        else
            A{k} = sigma(Z{k});
        end
    end
    NNx(j) = A{L}; 
end
end