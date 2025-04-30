function y = DiffSigmoid(x)
y = exp(x)./((exp(x)+1).^2);
end