READ ME: Vector Electric Field Reconstruction from Line of Sight (LOS) Radar Measurements Algorithm
Nicolls et al
translated by N. Maksimova (NAM)
readme created: August 27th, 2015, NAM
last edit: December 17th, 2015, NAM


The code in this directory implements the algorithm described in Nicolls, M. J, R. Cosgrove, and H. Bahcivan (2014), Estimating the vector electric field using monostatic, multibeam incoherent scatter radar measurements, Radio Sci., 49, 1124-1139, doi:10.1002/2014RS005519. For a theoretical treatment of the algorithm, please refer to the paper. 

Files included (alphabetical by kind):
MAT FILES
~azEl_to_geoMag.mat: contains conversion from azimuth+elevation to geomagnetic coordinates. DO NOT DELETE! this file takes hours to produce!
~coordconv.mat: contains coordinates for conductances 
~gradientsConductanceGrid.mat: contains saved gradient operators on the conductance grid computed in caseStudyFACcondGrid.py
~randor.mat: contains random numbers for simulating errors on velocity measurements consistently
~result_AMISR_FAC_Nov7_2012.mat: contains height-integrated conductances for case study event

PYTHON SCRIPTS
~caseStudyFACcondGrid.py: computes FACs for 11.07.2012 pfisr case study event using the grid on which the conductances were obtained (this script interpolates the fitter electric field results to the conductance grid)
~constants.py: global variables, including max error allowed for using data point, MAX ALTITUDE, etc
~generateGrid.py: create grid for fitter to use in computing field solution
~getData.py: either import real data or simulate data 
~getIokUnion.py: to be used for creating multiple fits for a long event (using more than one time record). finds intersection of all sets of “useable” data measurements so that the algorithm uses the same data set from time record to time record. Important! in order to use/disable the IokUnion set, go to main.py, to the comment that starts with “###IMPORTANT:” to set the variable useIokIntersection to 1/0 respectively. 
~getProjections.py: contains vector projection code (i.e. dot product, length(vector), angle(between vectors), etc) to be used for plotting E field in terms of its radial and angular components. 
~gradientsAndGMatrix.py: given a grid, creates gradient operators and the regularization G matrix
~hardCodeOverride.py: when the hardCodeOverride flag is turned on in main.py (in function runFitter()), the fitter accesses this script to read in the parameters of the runs the user wishes to have performed. otherwise, the user follows the terminal prompts to specify the parameters. 
~lostInTranslation.py: miscellaneous functions that didn’t have a neat MATLAB->Python translation or were easier to write from scratch. 
~main.py: main executable. loads optical data and either calculates fits or loads from existing files, calls plotter to plot optical + fit results. Important: make sure to set the variable in_dir to be the directory that contains the optical images you wish to plot.
~opticalPlotter.py: Steve’s code. main function is plot_allsky_FA(). plots optical data + FACs, fits, divergences, etc if desired. 
~plotProcess.py: creates standard figures for fit results (no optical). i.e. quiver plot of field, velocity plots and residuals, predicted/simulated fields+divergences, etc.
~resolveEfield.py: finds alpha parameter
~runSim.py: this script does most of the work. this is called from main.py and it goes through the process of calling the data loader, grid generator, operator creator, alpha solver functions, etc. 
~setUp.py: code for terminal prompter
~simPattern.py: simulates scalar potential and electric field


This program can be used to do two types of fits:
1) simulate an electric field, generate simulated radar LOS data for that field, and construct a fit using the theoretical algorithm. This allows the researcher to study the performance of the algorithm in detail, because the true E-field, divergence of the E-field, etc. are known (not the case with real data).
2) construct a fit for real radar LOS data. 

In addition, there are several files in this directory that are used to compute the FACs. 
caseStudyFACcondGrid.py contains the code that computes the FACs on the conductance grid
gradientsConductanceGrid.mat contains the gradient operators on the conductance grid (they are computed in caseStudyFACcondGrid.py and saved to this mat file since the computation is lengthy)
coordconv.mat contains the coordinates corresponding to the height-integrated conductances located in result_AMISR_FAC_Nov07_2012.mat






