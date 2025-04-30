%Het doel van dit bestand is een interface te hebben om makkelijk biases
%B en W te genereren
answer = questdlg('Is er al trainingdata aangemaakt?', ...
	'Trainingdata', ...
	'Ja','Nee','Nee');
switch answer
    case 'Ja'
        [filename, pathname] = uigetfile('*.mat', 'Selecteer een .mat file voor de trainingdata');

        if isequal(filename, 0)
            disp('Gebruiker heeft selectie geannuleerd.');
        else
            fullFilePath = fullfile(pathname, filename);
            loadedData = load(fullFilePath);
            disp('Trainingdata toegevoegd');
            disp(fieldnames(loadedData));
            TD = loadedData.TD
        end
    case 'Nee'
       f = input("Geef een functie waarvan trainingdata te generen:  ");
       a = input("Geef de ondergrens van het gesloten interval van de trainingdata: ");
       b = input("Geef de bovengrens van het gesloten interval van de trainingdata: ");
       N = input(['Geef het aantal equidistante punten in het interval [' num2str(a) ',' num2str(b) ']: ']);
       TD = TrainingData(f,a,b,N)
end

%Structuur van het netwerk bepalen

L_2 = input("Geef het aantal tussenlagen: ");
l_2 = zeros(1,L_2);
for j = 1:L_2
    l_2(j) = input(['Hoeveel neuronen zijn in laag ' num2str(j+1) '? ']);
end

%Willekeurige biases en gewichten toevoegen in het interval [0,1]
a = -1;
b = 1;
[B,W] = HeInitialization(l_2);
disp("Willekeurige biases:"),B
disp("Willekeurige gewichten:"),W

%Stochastische gradient descent

n = input("Geef de stapgrootte/learning rate van de methode: ");

N = input("Geef het aantal cycli voor de methode: ");