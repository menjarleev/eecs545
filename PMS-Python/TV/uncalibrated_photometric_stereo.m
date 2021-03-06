
% function [B,S] = uncalibrated_photometric_stereo(I,mask)
%
% An implementation of the uncalibrated photometric stereo technique described in 
% "Shape and Albedo from Multple Images using Integrability" by Yuille and Snow.
%
% I     : NxM Image data; N is the number of pixels per image, M is the number of images.
% mask : WxH Mask image to exclude pixels from being considered.
% B     : Nx3 Facet matrix (N=W*H). Each row corresponds to the surface normal times the albedo of the ith pixel.
% S     : 3xM Light source directions/intensities. The ith column corresponds to the light source
%         for the ith image. The direction of the vector specifies the light source direction and
%         the magnitude represents the light source intensity.
%
% ============
% Neil Alldrin / Yvain Queau
function [B,S] = uncalibrated_photometric_stereo(I,mask)

[nrows ncols]=size(mask);
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Gestion des données
%%%%%%%%%%%%%%%%%%%%%%%%%%

Isize = size(mask);
I = im2double(I); % I is required to be of type double for later operations to work
Iinds = find(mask(:)>0); % indices of non-masked pixels
M = size(I,2);          % # of images / illuminations
N = size(I,1);  % # of non-masked pixels

% Transpose I so that it's consistent with the yuille and snow paper
I_ = I'; % non-masked
% size(I_)
% Iinds
I = I_(:,Iinds); % masked


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SVD
%%%%%%%%%%%%%%%%%%%%%%%%%%

% get the 3 most significant singular values (we dont use the ambient term)
[f D e] = svds(I,3);
DftPlus=pinv(D*f');
e_ = I_' * DftPlus(:,1:3);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Estimation des cofacteurs (Yuille/Snow paper)
%%%%%%%%%%%%%%%%%%%%%%%%%%
% We now need to restrict f and e by enforcing integrability

%  compute de/dx and de/dy
smooth_kernel = fspecial('gaussian',[25 25],2);
for i = 1:3
  e_smooth = imfilter(reshape(e_(:,i).*mask(:),Isize),smooth_kernel,'same');
  [dedx_ dedy_] = gradient(e_smooth);
  dedy_ = -dedy_;
  dedx(:,i) = dedx_(Iinds);
  dedy(:,i) = dedy_(Iinds);
end

%  solve the following system of equations for x_ij and y_ij
%   sum_{j=2:3} sum_{i=1:j-1} ( a_ij*x_ij - b_ij*y_ij) = 0
%  where,
%   a_ij = e(i)*dedx(j)-e(j)*dedx(i)
%   c_ij = P_3i*P_2j - P_2i*P_3j
%   b_ij = e(i)*dedy(j)-e(j)*dedy(i)
%   d_ij = P_3i*P_1j - P_1i*P_3j

a12 = e(:,1).*dedx(:,2) - e(:,2).*dedx(:,1);
a13 = e(:,1).*dedx(:,3) - e(:,3).*dedx(:,1);
a23 = e(:,2).*dedx(:,3) - e(:,3).*dedx(:,2);

b12 = e(:,1).*dedy(:,2) - e(:,2).*dedy(:,1);
b13 = e(:,1).*dedy(:,3) - e(:,3).*dedy(:,1);
b23 = e(:,2).*dedy(:,3) - e(:,3).*dedy(:,2);

% Solve the system A*x = 0 
A = [a12 a13 a23 -b12 -b13 -b23 ];
[AU,AS,AV] = svd(A(:,1:6),0);
x = AV(:,size(AV,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GBR initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize the values of the GBR parameters by assuming the lightings 
% have constant magnitude (fixed to some acceptable value)
Sx=f(:,1:3)*D(1:3,1:3)*transpose([-x(3) x(2) -x(1)]);
Sy=f(:,1:3)*D(1:3,1:3)*transpose([x(6) -x(5) x(4)]);
Sz=f(:,1:3)*D(1:3,1:3)*transpose([0 0 1]);
B=Sx.*Sx+Sy.*Sy;
cste=(1+eps)*max(Sx.^2+Sy.^2+Sz.^2); % the magnitude of lightings
Asys=f(:,1:3)*D(1:3,1:3);
K=(sqrt(cste*ones(M,1)-B));
%% Résolution du systeme
x20=[pinv(Asys)*K;cste];
x2=x20;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calcul final de P et Q
%%%%%%%%%%%%%%%%%%%%%%%%%%
P3inv = [-x(3)  x(6)  x2(1); ...
	  x(2) -x(5)  x2(2); ...
	 -x(1)  x(4)  x2(3)];

%  invert P3inv to get P3 (up to GBR)
P3 = inv(P3inv);

%  solve for Q3 using the relation P3'*Q3 = D3
Q3 = (P3') \ D(1:3,1:3);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calcul final de B=\rho N et S
%%%%%%%%%%%%%%%%%%%%%%%%%%
B = e_(:,1:3)*P3';
S = Q3*f(:,1:3)';
% Adjust the sign so that nz is positive
if mean(B(find(mask(:),3)))<0
  B = B*[1 0 0; 0 1 0; 0 0 -1];
  S = [1 0 0; 0 1 0; 0 0 -1]*S;
  
end


end
