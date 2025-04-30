function y = LeakyReLU(x,a)
    y = max(x,a*x);
end