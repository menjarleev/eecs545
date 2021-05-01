function  [F G] = tv_M1(x,square_norm_gradient_M1,square_norm_gradient_M3,scalar_product,mask)

	[nrows ncols]=size(mask);	
	norm_gradient=sqrt(square_norm_gradient_M1+x*x*square_norm_gradient_M3+2*x*scalar_product);	
	mask_tv=(mask>0).*(norm_gradient>0);
	F=sum(norm_gradient(mask_tv>0));
	
	dx_norm_gradient=(x*square_norm_gradient_M3+scalar_product)./norm_gradient;
	G=sum(dx_norm_gradient(mask_tv>0));
end

