function [muTV nuTV]=resolutionGBRTVZ(N_psdI,mask)

	Ntilde=transpose(N_psdI);
	[nb_lignes ,nb_colonnes]=size(mask);
	exterieur_Beethoven=find(mask==0);
	% Estimation initiale aux moindres carrés
	M1=reshape(Ntilde(1,:,:),nb_lignes,nb_colonnes);
	M2=reshape(Ntilde(2,:,:),nb_lignes,nb_colonnes);
	M3=reshape(Ntilde(3,:,:),nb_lignes,nb_colonnes);	
	M1(exterieur_Beethoven)=0;
	M2(exterieur_Beethoven)=0;
	M3(exterieur_Beethoven)=0;
	masktvz=find((abs(M3)>0)&(mask>0));
	nbmasktvz=length(masktvz);
	mu=-mean(M1(masktvz)./M3(masktvz));
	nu=-mean(M2(masktvz)./M3(masktvz));
	x0=[mu nu];	
	options = optimset('Display','off','GradObj','on');	
	x = fminunc(@(x) tvz(x,M1,M2,M3,masktvz), x0,options);
	muTV=x(1);
	nuTV=x(2);

end
