%UPS_TV solves the Uncalibrated Photometric Stereo problem 
%	[N,RHO,L] = ups_tv(I,MASK,METHOD) uses the method METHOD to solve 
%	the UPS problem from the images stored in I, assuming only the 
%	white pixels of MASK are informative. 
%
%	It accompanies the methods described in the paper : 
%	Solving Uncalibrated Photometric Stereo Using Total Variation 
%	by Yvain Queau, Francois Lauze and Jean-Denis Durou (JMIV 2014)	
%
%	I must be a NxM 'double' matrix, where N is the number of pixels in 
%	each image and M is the number of images
%
%	MASK must be a NROWSxNCOLS matrix such that MASK(i,j)=0 if (i,j) is 
%	outside the mask, and MASK(i,j)>0 if (i,j) is inside the mask. Note 
%	that MASK must hold size(mask,1)*size(mask,2)=size(I,1)
%
%	METHOD must be an integer between 1 and 4 specifying which method 
%	is to be used : 
%		1	TV-M
%		2	TV-M (G.f)
%		3	TV-u
%		4	TV-u/ME
%
%	N is a Nx3 matrix containing the unitary normals at each pixel
%
%	RHO is a matrix of same size as MASK, containing the values of the 
%	albedo
%
%	L is the 3xM lighting matrix
%
%	Credits : 
%		This code is free to use for research purpose only. If 
%		you use the results from this code in a paper, please include a 
%		reference to our JMIV 2014 paper. 
%		
%		The method 4 (TV-u/ME) also partly uses 
%		the code from Neil Alldrin accompanying his CVPR 2007 paper 
%		'Resolving the Genearlized-Bas-Relief ambiguity by Entropy 
%		Minimization'. 	
function [n,rho,L]=ups_tv(I,mask,method)
	% Apply the SVD factorization of I described in Yuille and Snow 1997
	% Note that the initialization of I is not random (see code for detail)
	% so as to limit the values of lambda to be searched in the TV-u/ME 
	% method
	[M_init,L_init] = uncalibrated_photometric_stereo(I,mask);
	
	% Calculate the GBR parameters
	switch method
		case 1
			% Mu and Nu estimated by TV-M
			[mu nu]=resolutionGBRTV(M_init,mask);		
			% Lambda estimated using Uniform Intensity
			la=solve_br(L_init,mu,nu);
		case 2
			% Mu and Nu estimated by TV-M (G.f)
			[mu nu]=resolutionGBRTV_filtrage(M_init,mask);		
			% Lambda estimated using Uniform Intensity
			la=solve_br(L_init,mu,nu);
		case 3
			% Mu and Nu estimated by TV-u 
			[mu nu]=resolutionGBRTVZ(M_init,mask);		
			% Lambda estimated using Uniform Intensity
			la=solve_br(L_init,mu,nu);
		case 4
			% Mu and Nu estimated by TV-u 
			[mu nu]=resolutionGBRTVZ(M_init,mask);
            
%             mu = .131; %.131
%             nu = .1; %.0339;
            
			% Lambda estimated using Min-Entropy
			M_tv=M_init*[1 0 0; 0 1 0; mu nu 1];
			la = solve_gbr2(M_tv);
		otherwise
			disp('Wrong value for METHOD')
	end
	
	% Apply GBR 
	M=M_init*[1 0 0; 0 1 0; mu nu la];
	rho=sqrt(sum(M.^2,2));
	pb = find(rho==0);
	n=M./repmat(rho,[1 3]);
	n(pb,1:2)=0;
	n(pb,3)=1;
	rho(pb) = 0;
	rho=reshape(rho,size(mask));
	L=[1 0 0; 0 1 0; -mu/la -nu/la 1/la]*L_init;	
	if(mean(n)<0)
		n=-n;
		L=-L;
	end	
	n(mask==0,:)=0;
	rho(mask==0)=0;

end
