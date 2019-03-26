clear
load('dataonly.txt');
%x = dataonly(:,3:8); 
%y = dataonly(:,9:16);
x = transpose(dataonly(:,1:19));
y = transpose(dataonly(:,20));

% vytvorenie štruktúry NS

pocet_neuronov=10;
net=patternnet(pocet_neuronov,'trainlm');
 
% % vyber rozdelenia
net.divideFcn='dividerand'; % náhodné rozdelenie
 
%net.divideFcn='divideblock';% blokove
 
%net.divideFcn='divideint';  % kazdy n-ta vzorka
 
%net.divideFcn='dividetrain';  % iba trenovacie

net.divideParam.trainRatio=0.6;
net.divideParam.valRatio=0;
net.divideParam.testRatio=0.4;

% Nastavenie parametrov trénovania
net.trainParam.goal = 1e-10;     % Ukoncovacia podmienka na chybu
net.trainParam.show = 5;        % Frekvencia zobrazovania priebehu chyby trénovania net.trainParam.epochs = 100;  % Max. po?et trénovacích cyklov.
net.trainParam.epochs = 1000;    % maximalny pocet trenovacich epoch.
 
% Trénovanie NS
net=train(net,x,y);
 
% Simulácia výstupu NS
outnetsim = sim(net,x);
 
% vypocet chyby siete
SSE=sum((y-outnetsim).^2);
MSE=SSE/length(y);
 
% Vykreslenie priebehov 

figure
plotconfusion(y,outnetsim);