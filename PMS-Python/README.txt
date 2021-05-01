This archive contains the Matlab .m files corresponding to the methods described in our JMIV 2014 paper (currently under review). Note that previous results with less explanations are available in our SSVM 2013 paper.

*****************************
*****	Copyright	*****
*****************************

1) The codes implementing the resolution of the GBR using TV are free to use for research purpose only. If you use the results from these codes in a paper, please include a reference to the related paper(s) : 
*****
Solving Uncalibrated Photometric Stereo using Total Variation 
Yvain Quéau, François Lauze, Jean-Denis Durou
JMIV 2014 - To appear, preprint available at http://queau.perso.enseeiht.fr
*****
Solving the Uncalibrated Photometric Stereo problem using Total Variation 
Yvain Quéau, François Lauze, Jean-Denis Durou
SSVM 2013
*****



The provided codes may use methods described in other papers. Depending on your usage, please be aware that the corresponding methods are described in the following papers : 

2) The method 4 (TV-u/ME) partly uses the method described in 
*****
Resolving the Generalized Bas-Relief Ambiguity by Entropy Minimization
Neil Alldrin, Satya Mallick, David Kriegman
CVPR 2007
*****

3) The initialization is a variant of the method of Yuille and Snow described in 
*****
Shape and Albedo from Multiple Images using Integrability
A. Yuille and D. Snow
CVPR 1997
*****

4) The preprocessing part is described in 
*****
Robust Photometric Stereo via Low-Rank Matrix Completion and Recovery
L Wu, A Ganesh, B Shi, Y Matsushita, Y Wang, Y Ma
ACCV 2010
*****

5) The dataset 'Beethoven.mat' provided contains 3 images, a mask and the lightings. It was created using the data available at : 
*****
http://www.ece.ncsu.edu/imaging/Archives/ImageDatabase/
*****

6) The code for integrating the normal field into a 3D-shape is a least square sparse solver on a non-rectangular area, described in
*****
Integration d'un champ de gradient rapide et robuste aux discontinuites - Application a la stereophotometrie
Queau, Y and Durou, J.-D.
RFIA 2014
*****


*********************
*****	Files	*****
*********************
This archive contains :
_ a dataset Beethoven.mat containing 3 images stored in a matrix I, a mask and the ground truth light matrix
_ a demo script demo_tv.m
_ a function for integrating the normal field direct_weighted_poinsson.m
_ the files for preprocessing the images in the folder Preprocessing
_ the files for solving the GBR using TV in the folder TV

*********************
*****	Usage	*****
*********************
A demo script using the Beethoven dataset is provided in demo_tv.m 

The main function is ups_tv.m in the folder TV . Given several images and a mask, it computes the normal field, the albedo map and the light matrix. 

Note that the normal field (and thus the light matrix) is estimated only up to a concave/convex ambiguity, so the integrated surface might occasionally not be well-oriented. 

It first calls uncalibrated_photometric_stereo to apply the method of Yuille and Snow for initializing the M-field and the lighting matrix
Then, it estimates mu and nu by calling one of the functions resolutionGBRTV***.m (3 functions corresponding to the 3 proposed methods)
Finally, it estimates lambda by calling either solve_br.m (constant magnitud of lightings) or solve_gbr2 (min entropy)

If you are not sure which method to use, you should probably use the method 4 (TV-u/me). It is not as fast as the others, but it will always give a somehow acceptable result.

*****************************
*****	Datasets	*****
*****************************
For copyright reasons, we cannot provide in this archive the other datasets used in the experiments of our JMIV 2014 paper. If you are interested in obtaining the corresponding .mat files, feel free to ask them by sending looking at their authors website


