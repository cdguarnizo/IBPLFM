function plotpred(tit, uTest, xTest, varTest, yTrain, xTrain, yTest)

if ~iscell(xTest),
    np = size(uTest,1);
    xT = cell(np,1);
   for d=1:np,
       xT{d} = xTest(:);
   end
   xTest = xT;
end

if nargin > 6,
    if ~iscell(xTest),
        np = size(uTest,1);
        xT = cell(np,1);
        for d=1:np,
            xT{d} = xTrain(:);
        end
        xTrain = xT;
    end
end

markerSize = 20;
markerWidth = 6;
markerType = 'k.';
lineWidth = 2;
fillColor = [0.8 0.8 0.8];
for d = 1:length(uTest),
    hold off;
    figure(d);
    clf
    ySd = sqrt(varTest{d});
    yPred = uTest{d};
    xT = xTest{d};
    fill([xT; xT(end:-1:1)], ...
         [yPred; yPred(end:-1:1)] ...
         + 2*[ySd; -ySd], ...
         fillColor,'EdgeColor',fillColor)
    hold on;
    h = plot(xT, yPred, 'k-');
    set(h, 'linewidth', lineWidth)
    if nargin > 4,
        p = plot(xTrain{d}, yTrain{d}, markerType);
        set(p, 'markersize', markerSize, 'lineWidth', markerWidth);
    end
    
    if nargin > 6,
        p = plot(xTest{d}, yTest{d}, 'ok');
        set(p, 'markersize', 6, 'lineWidth', 2);
    end
    title(strcat('Prediction ',tit,' ',num2str(d)));
end