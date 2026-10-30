function [p, yaw] = quad_reference(t, name)
s = max(0, t-3);
ramp = 3;
w = 0.35;
radius = 1.2;
r = min(s/ramp, 1);

if s < ramp
    phase = w*ramp*(2.5*r^4-3*r^5+r^6);
else
    phase = w*(s-ramp/2);
end
yaw = 0.35*sin(phase/2);

switch name
    case 'hover'
        p = [0;0;1];
        yaw = 0;

    case 'circle'
        p = [radius*(1-cos(phase));radius*sin(phase);1+0.25*(1-cos(phase))];

    case 'figure8'
        p = [radius*sin(phase);0.65*radius*sin(2*phase);1+0.25*(1-cos(phase))];

    case 'waypoints'
        points = [0 0 1;1 0 1.6;1 1 1.6;-1 1 1.2;0 0 1];
        starts = [2 8 14 20];
        p = points(1,:)';
        for j = 1:4
            r = max(0, min(1, (t-starts(j))/4));
            p = p+(10*r^3-15*r^4+6*r^5)*(points(j+1,:)-points(j,:))';
        end

    otherwise
        error('Unknown trajectory.');
end
end
