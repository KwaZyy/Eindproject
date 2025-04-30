function y = DiffLeakyReLU(x,a)
if x >= 0
    y = 1;
else
    y = a;
end
end