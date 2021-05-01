function  [F G] = tvz(MuNu,M1,M2,M3,mask)
%%%%%%
%%%	MuNu: 	(mu,nu) les deux parametres à estimer
%%%	DM	:	les derivees de rhoN : in R^{6*nb_lignes*nb_colonnes}
%%%%%%

	
	mu=MuNu(1);
	nu=MuNu(2);

	M1=M1(mask);
	M2=M2(mask);
	M3=M3(mask);
	
	% Fonction objectif		
	U1=-M1./M3-mu;
	U2=-M2./M3-nu;
	racU12plusU22=sqrt(U1.^2+U2.^2);
	F=sum(racU12plusU22);
	%~ F=sum(U1.^2+U2.^2);
	ind=find(racU12plusU22>0);
	% Gradient
	G=zeros(2,1);
	%~ G(1)=-sum(U1./racU12plusU22);
	%~ G(2)=-sum(U2./racU12plusU22);
	G(1)=-sum(U1(ind)./racU12plusU22(ind));
	G(2)=-sum(U2(ind)./racU12plusU22(ind));
	%~ G(1)=-2*sum(U1(:));
	%~ G(2)=-2*sum(U2(:));
end

