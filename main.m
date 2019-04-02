
clear
load('dataonly.txt');
%x = dataonly(:,3:8); 
%y = dataonly(:,9:16);
x = transpose(dataonly(:,1:19));
y = transpose(dataonly(:,20));

% vytvorenie štruktúry NS

pocet_neuronov=18;
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
net.trainParam.goal = 1e-7;     % Ukoncovacia podmienka na chybu
net.trainParam.show = 5;        % Frekvencia zobrazovania priebehu chyby trénovania net.trainParam.epochs = 100;  % Max. po?et trénovacích cyklov.
net.trainParam.epochs = 1500;   % maximalny pocet trenovacich epoch.
net.trainParam.min_grad = 1e-8;

for i=1:10
    net=patternnet(pocet_neuronov,'trainlm');
    net.divideFcn='dividerand';
    net.divideParam.trainRatio=0.6;
    net.divideParam.valRatio=0;
    net.divideParam.testRatio=0.4;
    net.trainParam.goal = 1e-7;     % Ukoncovacia podmienka na chybu
    net.trainParam.show = 5;        % Frekvencia zobrazovania priebehu chyby trénovania net.trainParam.epochs = 100;  % Max. po?et trénovacích cyklov.
    net.trainParam.epochs = 1500;   % maximalny pocet trenovacich epoch.
    net.trainParam.min_grad = 1e-8;

% Trénovanie NS
    net=train(net,x,y);
 
% Simulácia výstupu NS
    outnetsim = sim(net,x);
 
% vypocet chyby siete
    SSE=sum((y-outnetsim).^2);
    MSE=SSE/length(y);
 
% [c,cm,ind,per] = confusion(y, outnetsim);
% Vykreslenie priebehov 

    figure(i)
    plotconfusion(y,outnetsim);
    pause(0.5);
end

testVector = randi(1151,10,1);          % vyber 10 nahodnych cisel z rozsahu 1-1151
testInput = transpose(dataonly(testVector,1:19));   %vyber vstupnych dat podla nahodnych indexov
testOutput = transpose(dataonly(testVector,20));    %vyber vystupnych dat

test = sim(net,testInput);                          % testovanie 10 nahodnych vzoriek
%vec2 = vec2ind(sim,10);
figure(11);
for i=1:10
    plotconfusion(testOutput,test); %confusion matica nahodnej vzorky
end
