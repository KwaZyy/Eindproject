%Dit bestand genereert trainingdata voor het 2D Poisson probleem
m = input("m: ");
tic
h = 1/(m+1);

%Sensoren genereren
[sensor_int, sensor_ext] = SensorGenerator(m);

og = -2; %ondergrens
bg = 2; %bovengrens
l = input("aantal trainingdata exemplaren: ");

%Hierin behouden we informatie van de discretisatie van f en g (per rij)
xF = zeros(l,m.^2);
xG = zeros(l,4.*m+4);
yF = xF;
yG = xF;

%(0,1) x (0,1) vierkant geometrie
gdm = [3 4 0 1 1 0 0 0 1 1]';
geom = decsg(gdm, 'S1', ('S1')');


for k = 1:l
    r = og + (bg-og).*rand(1,9);
    %Willekeurige tweedegraadspolynomen voor R^2
    f = @(x,y) r(1).*x.^2.*y.^2 + r(2).*x.^2.*y + r(3).*x.*y.^2 + r(4).*x.*y + r(5).*x.^2 ...
        + r(6).*y.^2 + r(7).*x + r(8).*y + r(9);
    g = f; 
    xf = DiscretizeFunc(f,sensor_int);
    xg = DiscretizeFunc(g,sensor_ext);

    %dit is nu de werkelijke inputdata van het branchnetwerk
    xF(k,:) = xf;
    xG(k,:) = xg;

    %We creëren nu de correcte uitvoer van de trainingdata via de PDE
    %toolbox in Matlab
    modelf = createpde();
    modelg = createpde();

    geometryFromEdges(modelf, geom);
    geometryFromEdges(modelg, geom);

    f_pde = @(location,state) f(location.x, location.y);
    g_pde = @(location,state) f(location.x, location.y);

    specifyCoefficients(modelf, 'm', 0, 'd', 0, ...
                    'c', 1, 'a', 0, 'f', f_pde); %c = 1 omdat -\delta u = f
    specifyCoefficients(modelg, 'm', 0, 'd', 0, ...
                    'c', 1, 'a', 0, 'f', 0);
    applyBoundaryCondition(modelf, 'dirichlet', ...
                       'Edge', 1:4, 'u', 0);
    applyBoundaryCondition(modelg, 'dirichlet', ...
                       'Edge', 1:4, 'u', g_pde);

    generateMesh(modelf, 'Hmax', 0.05);
    generateMesh(modelg, 'Hmax', 0.05);

    resultsf = solvepde(modelf);
    resultsg = solvepde(modelg);
    uf = resultsf.NodalSolution;
    ug = resultsg.NodalSolution;

    %y output voor sensorwaarden (dit moet niet noodzakelijk de sensoren
    %zijn, eenderwelke collectie van punten die goed over het vierkant
    %spannen zullen werken)
    yf = interpolateSolution(resultsf, sensor_int(:,1), sensor_int(:,2));
    yg = interpolateSolution(resultsg, sensor_int(:,1), sensor_int(:,2));

    yF(k,:) = yf;
    yG(k,:) = yg;
end
toc
save('sensor_int.mat','sensor_int')
save('sensor_ext.mat','sensor_ext')

save('xF.mat', 'xF');
save('yF.mat', 'yF');

save('xG.mat', 'xG');
save('yG.mat', 'yG');