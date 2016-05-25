function error = errorMeasureRegress(mustar, varstar, y, type, Ytrain)


N = size(y,1);
res = mustar-y;

switch type 
    case 'mse'
        error = sum(res.^2)/N;
    case 'smse'
        error = sum(res.^2)/N;
        error = error/var(y,1); 
    case 'nlp' 
        error = 0.5*mean(log(2*pi*varstar)+res.^2./varstar);
    case 'snlp'
        nlp = 0.5*mean(log(2*pi*varstar)+res.^2./varstar);
        muY  = mean(Ytrain);
        varY = var(Ytrain,1);
        error = nlp - 0.5*mean(log(2*pi*varY) + ((y-muY).^2)/varY);  
end