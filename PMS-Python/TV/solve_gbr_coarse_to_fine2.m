% Performs a coarse-to-fine search over a 3D grid of (mu,nu,lambda) values.
%
% ============
% Neil Alldrin
%
function [lambda] = solve_gbr_coarse_to_fine2(cost_function,lb,ub,stepSize,tol)

if ~exist('stepSize') stepSize = (ub-lb)/20; end;
if ~exist('tol') tol = stepSize/5; end;

lb_orig = lb; ub_orig = ub;

stepRatio = stepSize./(ub-lb);
e = inf; lambda=1;
while true
  
  %~ fprintf('\n **stepSize = [%f]**\n',stepSize);
  %~ fprintf(' **lb = [%f]**\n',lb);
  %~ fprintf(' **ub = [%f]**\n',ub);
  
  % perform brute-force search
  [lambda_] = solve_gbr_bruteforce2(cost_function,lb,ub,stepSize);
  e_ = cost_function([lambda_]);
  if (e_ < e)
    e = e_; lambda = lambda_;
  end
  
  if (all(stepSize <= tol)) break; end;
  
  % set new boundaries
  lb = [lambda]-1.5*stepSize;
  ub = [lambda]+1.5*stepSize;
  stepSize = stepSize/1.5;
  %stepSize = stepSize/1.5;
  %stepSize = stepRatio.*(ub-lb);

  lb = max(lb,lb_orig);
  ub = min(ub,ub_orig);
end
