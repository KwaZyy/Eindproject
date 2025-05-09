function [sensor_f, sensor_g] = SensorGenerator(m)
h = 1/(m+1);
sensor_f = zeros(m.^2,2);
sensor_g = zeros(4*m+4,2);
for i = 1:m
    for j = 1:m
        a = i+(j-1)*m; %index
        sensor_f(a,:) = [i*h; j*h];
    end
end
for i = 1:m
    a1 = i+1; a2 = m+2+i; a3 = 2*m+3+i; a4 = 3*m+4+i;
    sensor_g(a1,:) = [i*h; 0];
    sensor_g(a2,:) = [1; i*h];
    sensor_g(a3,:) = [1-i*h; 1];
    sensor_g(a4,:) = [0; 1-i*h];
end
sensor_g(1,:) = [0; 0];
sensor_g(m+2,:) = [1; 0];
sensor_g(2*m+3,:) = [1; 1];
sensor_g(3*m+4,:) = [0; 1];
end