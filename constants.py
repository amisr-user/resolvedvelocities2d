from math import *
from decimal import *

# CONSTANTS # 

e2 = 0.00669437999014; #first eccentricity squared 
a = 6378137.0; #earth's radius (m)

# defined parameters #

minAlt = 175; maxAlt = 600;# km ##was 700
#maxDv = 300.0; maxV = 2000.0; # m/s
#RHV raised these
#maxDv = 500.0; maxV = 3000.0; # m/s
#Lower values
maxDv = 300.0; maxV = 2000.0
minSNR=0.1
maxEl = 90.0;
BabsFixed = 50000e-9; # Tesla

