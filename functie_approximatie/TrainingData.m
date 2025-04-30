%Genereert trainingdata van een bepaalde functie via linspace. De
%trainingdata kan ook zelf opgeslaan worden

    %INPUT:
    %f: Functie waarvan trainingdata te genereren
    %a: Ondergrens interval
    %b: Bovengrens interval
    %N: Aantal equidistante punten

    %OUTPUT:
    %TD: Trainingdata
function TD = TrainingData(f,a,b,N)
x = linspace(a,b,N)';
TD = table(x,f(x),'VariableNames', {'x', 'y'});
uisave({'TD'},'TrainingData')
end