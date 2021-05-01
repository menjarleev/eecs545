% Performs a brute-force discrete search over a 3D grid of (mu,nu,lambda) values.
%
% ============
% Neil Alldrin
%
function [lambda] = solve_gbr_bruteforce2(cost_function,lb,ub,stepSize)

if ~exist('stepSize') stepSize = (ub-lb)/20; end;

%~ mus =     lb(1):stepSize(1):ub(1);
%~ nus =     lb(2):stepSize(2):ub(2);
lambdas = lb:stepSize:ub;

e_min = inf;
i_min = 1; j_min = 1; k_min = 1;
%~ for i = 1:length(mus)
  %~ for j = 1:length(nus)
    for k = 1:length(lambdas)
      
      lambda = lambdas(k);
      e = cost_function([lambda]);
    
      % check if this is the best found so far
      if any( e < e_min )
	e_min = e;  k_min = k;
	%~ fprintf('entropy=%f, mu=%f,nu=%f,l=%f\n',e_min,mus(i_min),nus(j_min),lambdas(k_min));
      end
    
    end % for i
  %~ end % for j
%~ end % for k

%~ mu = mus(i_min);
%~ nu = nus(j_min);
lambda = lambdas(k_min);
