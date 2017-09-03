function savemodel(model,con)

save(strcat('temp/m',num2str(con),'.mat'),'model');
end