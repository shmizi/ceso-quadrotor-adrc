function out = quad_math(operation, value)
switch operation
    case 'wrap'
        % value: angle or array of angles in rad, returned in the range [-pi, pi)
        out = mod(value+pi, 2*pi)-pi;

    case 'rotation'
        % value: orientation quaternion [w x y z]
        % out: 3x3 matrix that turns body-frame vectors into world-frame vectors
        q = value(:)/norm(value);
        w = q(1);
        x = q(2);
        y = q(3);
        z = q(4);
        out = [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w); ...
               2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w); ...
               2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)];

    case 'euler'
        % value: orientation quaternion [w x y z]
        % out: [roll; pitch; yaw] in rad (ZYX order), read off the rotation matrix R
        R = quad_math('rotation', value);
        out = [atan2(R(3,2), R(3,3)); asin(max(-1, min(1, -R(3,1)))); atan2(R(2,1), R(1,1))];

    case 'rate_map'
        % value: Euler angles [roll phi; pitch theta; yaw]
        % out: E, so that Euler-angle rates = E * body angular rates [p; q; r]
        % s, c: sine and cosine of roll; ct: cosine of pitch
        phi = value(1);
        theta = value(2);
        s = sin(phi);
        c = cos(phi);
        ct = cos(theta);

        % E divides by cos(pitch), so stop before pitch gets near 90 degrees
        if abs(ct) < 0.15
            error('quad:Attitude', 'Euler controller left moderate-tilt envelope.');
        end

        out = [1, s*tan(theta), c*tan(theta); ...
               0, c, -s; ...
               0, s/ct, c/ct];

    otherwise
        error('Unknown operation.');
end
end
