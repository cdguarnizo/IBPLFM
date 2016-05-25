function plotpredout(model, xTest)

if isfield(model,'outX'),
    [ymean yvar]=ibpmultigpPosterior(model, xTest);
    delta = 0;
    yTrain = model.y;
    xTrain = model.outX;
else
    [ymean yvar]=multigpPosteriorMeanVar(model, xTest{1});
    delta = model.nlf;
    yTrain = cell(1, model.nout);
    ini = 1;
    for d=1:model.nout,
        lX = length(model.X{d+delta})-1;
        yTrain{d} = model.y(ini:ini+lX);
        ini = ini + lX + 1;
    end
end

markerSize = 20;
markerWidth = 6;
markerType = 'k.';
lineWidth = 2;
fillColor = [0.8 0.8 0.8];
xTest = xTest{1};
for d = 1:model.nout,
    hold off;
    figure;
    clf
    ySd = sqrt(yvar{d+delta});
    yPred = ymean{d+delta};
    fill([xTest; xTest(end:-1:1)], ...
         [yPred; yPred(end:-1:1)] ...
         + 2*[ySd; -ySd], ...
         fillColor,'EdgeColor',fillColor)
    hold on;
    h = plot(xTest, yPred, 'k-');
    set(h, 'linewidth', lineWidth)
    p = plot(model.outX{d}, model.y{d}, markerType);
    set(p, 'markersize', markerSize, 'lineWidth', markerWidth);
    title(strcat('Prediction Output',num2str(d)))
end