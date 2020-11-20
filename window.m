%% window.m
%% Joseph Patton
%% 4-16-20

clear;
beta = 7;
N = 8192;
x = kaiser(N,beta);

fprintf("wind = [",N);
for i = 0:1023
    ind = i*8;
    fprintf("    %f, %f, %f, %f, %f, %f, %f, %f,\n",...
        x(ind+1),x(ind+2),x(ind+3),x(ind+4),...
        x(ind+5),x(ind+6),x(ind+7),x(ind+8));
end
ind = 1023*8;
fprintf("    %f, %f, %f, %f, %f, %f, %f, %f]\n",...
    x(ind+1),x(ind+2),x(ind+3),x(ind+4),...
    x(ind+5),x(ind+6),x(ind+7),x(ind+8));
