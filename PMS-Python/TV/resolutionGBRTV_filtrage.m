function [muTV nuTV laTV]=resolutionGBRTV(N_psdI,mask)

	N=transpose(N_psdI);
	[nb_lignes nb_colonnes]=size(mask);	
	exterieur_Beethoven=find(mask==0);
	%%%%%
	%%%	Etape 3 : Variation totale (GBR)
	%%%%%

	% Estimation initiale aux moindres carrés
	Ntilde=N;
	M1=reshape(Ntilde(1,:,:),nb_lignes,nb_colonnes);
	M2=reshape(Ntilde(2,:,:),nb_lignes,nb_colonnes);
	M3=reshape(Ntilde(3,:,:),nb_lignes,nb_colonnes);	
	M1(exterieur_Beethoven)=0;
	M2(exterieur_Beethoven)=0;
	M3(exterieur_Beethoven)=0;
	smooth_kernel = fspecial('gaussian',[25 25],2);	
	M1=imfilter(M1.*mask,smooth_kernel,'same');
	M2=imfilter(M2.*mask,smooth_kernel,'same');
	M3=imfilter(M3.*mask,smooth_kernel,'same');
	[DM1DX DM1DY]=gradient(M1);
	[DM2DX DM2DY]=gradient(M2);
	[DM3DX DM3DY]=gradient(M3);
	DM1DX(exterieur_Beethoven)=0;
	DM1DY(exterieur_Beethoven)=0;
	DM2DX(exterieur_Beethoven)=0;
	DM2DY(exterieur_Beethoven)=0;
	DM3DX(exterieur_Beethoven)=0;
	DM3DY(exterieur_Beethoven)=0;	
	square_norm_gradient_M1=DM1DX.*DM1DX+DM1DY.*DM1DY;
	square_norm_gradient_M2=DM2DX.*DM2DX+DM2DY.*DM2DY;
	square_norm_gradient_M3=DM3DX.*DM3DX+DM3DY.*DM3DY;
	scalar_product_M1_M3=DM1DX.*DM3DX+DM1DY.*DM3DY;
	scalar_product_M2_M3=DM2DX.*DM3DX+DM2DY.*DM3DY;		
	% Tikhonov initialization
	sum_square_norm_gradient_M3=sum(square_norm_gradient_M3(mask>0));
	mu=-sum(scalar_product_M1_M3(mask>0))/sum_square_norm_gradient_M3;
	nu=-sum(scalar_product_M2_M3(mask>0))/sum_square_norm_gradient_M3;	
	% TV minimization
	options = optimset('GradObj','on','Display','off');
	muM1=fminunc(@(x) tv_M1(x,square_norm_gradient_M1,square_norm_gradient_M3,scalar_product_M1_M3,mask),mu,options);
	nuM2=fminunc(@(x) tv_M1(x,square_norm_gradient_M2,square_norm_gradient_M3,scalar_product_M2_M3,mask),nu,options);
	muTV=muM1;
	nuTV=nuM2;	
	laTV=1;
	

end

