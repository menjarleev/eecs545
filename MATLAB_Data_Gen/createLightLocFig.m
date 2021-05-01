function createLightLocFig(X1, Y1)
%CREATEFIGURE(X1, Y1)
%  X1:  scatter x
%  Y1:  scatter y

%  Auto-generated by MATLAB on 30-Apr-2021 00:53:53

% Create figure
figure('Color',[1 1 1]);

% Create axes
axes1 = axes;
hold(axes1,'on');

% Create scatter
plot(X1,Y1, 'o', 'MarkerFaceColor',[0 0.450980392156863 0.741176470588235],...
    'LineWidth',8, 'MarkerSize', 10);

% Create ylabel
ylabel({'Vertical Location'});

% Create xlabel
xlabel({'Horizontal Location'});

% Create title
title({'Light Locations Relative to Camera'});

% Uncomment the following line to preserve the X-limits of the axes
xlim(axes1,[-2 2]);
% Uncomment the following line to preserve the Y-limits of the axes
ylim(axes1,[-1.5 1.5]);
box(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontSize',20,'XGrid','on','YGrid','on');