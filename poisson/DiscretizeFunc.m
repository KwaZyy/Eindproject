function xh = DiscretizeFunc(h,sensor)
l = length(sensor);
xh = zeros(l,1);
for i = 1:l
    x = sensor(i,1);
    y = sensor(i,2);
    xh(i) = h(x,y);
end
end

