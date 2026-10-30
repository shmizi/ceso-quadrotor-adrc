function quad_capture(fig, filename)
drawnow;

if ~isMATLABReleaseOlderThan('R2025a')
    exportapp(fig, filename);
else
    visibility = fig.Visible;
    cleanup = onCleanup(@() restore_visibility(fig, visibility));
    fig.Visible = 'on';
    drawnow;
    frame = getframe(fig);
    imwrite(frame.cdata, filename);
end
end


function restore_visibility(fig, visibility)
if isgraphics(fig)
    fig.Visible = visibility;
end
end
