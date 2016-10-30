Directory: mkpict
=================

### DIRECTORIES:

- mkpict -- contains functions and Matlabs scripts that are used to make the
   Matlab-based plots appearing in the book.  These are provided for
   researchers who want to profit from the form of a plot for their
   own work, and who want to test the result.

   Correspondance between the figures and the Matlab files is provided
   by the file mkpict/listoffigures.

   Many of these functions and scripts are were used as the basis for
   testing the algorithms referenced in the book, and in that regard
   should be regarded as part of the testing suite.

-------------------------------------------------------

### TOOLBOXES:
The code written should run with Matlab without additional toolboxes,
with the following known exceptions:

`mkpict/testeigfil.m` and `mkpict/testeigfil2.m` use the `freqz` function for plotting (found in the signal processing toolbox)

***************************************************************
Directory: mkpict
***************************************************************
```matlab
attract1.m
% a plot showing an attractor
```

```matlab
attract2.m
% a plot showing an attractor
```

```matlab
bayes1.m
% Bayes decision tests
```

```matlab
bayes2.m
% Bayes decision tests for Gaussian
```

```matlab
bayes4.m
% show the decision regions for a 3-way test
```

```matlab
binchan.m
% 
% Data for Bayesian detection on the binary channel
```

```matlab
binchanex.m
% data for binary channel
```

```matlab
chebyplot.m
% Plot Chebyshev polynomials
```

```matlab
chi2plot.m
% 
% Plot the chi-squared r.v.
```

```matlab
compmap3.m
% make figure comppos1
```

```matlab
condhilb.m
% Plot the condition of the Hilbert matrix
```

```matlab
drawtrellis.m
% 
% Draw a trellis in LaTeX picture mode
%
% function drawtrellis(fid,numbranch,r,p)
%
% fid = output file id
% numbranch = number of branches to draw
% r = path cost values
% p = flag
%
% Other values are contained in global variables.  See the file
```

```matlab
drawtrelpiece.m
%
% Draw a piece of a trellis in LaTeX picture mode
%
% fname = file name
% trellis = trellis description
% branchweight = weights of branches
```

```matlab
drawvit.m
% Program to draw the paths for the Viterbi algorithm using a LaTeX picture
```

```matlab
duality1.m
% Make a plot illustrating duality
```

```matlab
eigdir.m
% make a contour plot of eigenstuff
```

```matlab
eigdir2.m
% make a contour plot of eigenstuff
```

```matlab
eigdirex.m
% make a contour plot of eigenstuff
```

```matlab
eigdist.m
% show the asymptotic equal distribution of eigenvalues
```

```matlab
ellipse.m
% Plot contours of an ellipse with large eigenvalue disparity
% and the results of steepest descent
```

```matlab
ellipsecg.m
% Plot contours of an ellipse with large eigenvalue disparity
% and the results of conjugate gradient.
```

```matlab
entplot.m
% plot the binary entropy function
```

```matlab
expmod.m
% Test Cadzow's results on the sinusoidal modeling
```

```matlab
fourser.m
% example Fourier series
```

```matlab
hilb1.m
% Program to generate the data for the hilbert approximation to
% the exponential function
```

```matlab
ifs3.m
% Plot the logistic map and the orbit of a point
```

```matlab
ifs3b.m
% Plot the logistic map and the orbit of a point
% do not specify lambda and x0 here: it is done by an upper script
```

```matlab
ifs4.m
% Demonstrate the logistic map
```

```matlab
ifsex3.m
% find an affine transformation Ax + b that transforms from
% {x00,x10,x20} to {x01,x11,x21}
```

```matlab
ifsfig1.m
% Make side-by-side figures
```

```matlab
legendreplot.m
% Plot legendre polynomials
```

```matlab
makeim.m
% make a test image for tomography example
```

```matlab
matcond.m
% Make an ill-conditioned matrix of sinusoids.
```

```matlab
matcond2.m
% Set up an ill-conditioned matrix of sinusoids
```

```matlab
min1.m
% make the contour plot for wftest
```

```matlab
min2.m
% make the contour plot for wftest
```

```matlab
moveiter.m
% test the solution of a moving RHS in the equation Ax=b
```

```matlab
newt1.m
% Demonstrate newton's stuff
```

```matlab
newt2.m
% Demonstrate newton's stuff on Rosenbrocks function
```

```matlab
oddeven.m
% data for odd/even game
```

```matlab
orthog.mma
(* sample file for orthogonalization *)
```

```matlab
patrec1.m
% generate some simple pattern recognition example data
```

```matlab
plotI0.m
% Plot the Bessel function
```

```matlab
plotJsurf.m
% plot a quadratice error surface
```

```matlab
plotbpsk.m
% Plot the probability of error for BPSK
```

```matlab
plotgauss.m
% Plot the Gaussian function
```

```matlab
plotgauss2.m
% Plot approximations to the central limit theorem
```

```matlab
plotgauss3.m
% plot a Gauss surface plot
```

```matlab
plotwavelet.m
% plot the wavelet data
```

```matlab
roc1.m
% plot the roc for a gaussian r.v.
```

```matlab
roc2.m
% plot the roc for a a xi^2
```

```matlab
roc3.m
% plot the roc for a gaussian r.v. and its conjugate
```

```matlab
rosenbrock.m
% Plot the Rosenbrock function contours
```

```matlab
rosengrad.m
% 
% compute the gradient of the rosenbrock function for test purposes
% function grad = rosengrad(x)
```

```matlab
saddle1.m
% make a saddle plot 
```

```matlab
scatter.m
% create a scatter plot to demonstrate principal component
```

```matlab
scatterex.m
% create a scatter plot to demonstrate principal component
```

```matlab
sigmoid.m
% plot the sigmoid function
```

```matlab
steeperr.m
% Plot errors of the steepest descent
```

```matlab
steeperrplot.m
% Make plots of error for steepest descent
```

```matlab
steepest1.m
% Demonstrate steepest descent on Rosenbrocks function
```

```matlab
sugitest.m
% test the Sugiyama algorithm
```

```matlab
surf1.m
% make a surface plot
```

```matlab
test2regress.m
% Test the formulas for regression in two dimensions
% input: x and y vectors
```

```matlab
test2regress2.m
% Test the formulas for regression in two dimensions
% input: x and y vectors
```

```matlab
testeigfil.m
% Test the eigenfilter stuff
```

```matlab
testeigfil2.m
% Test the eigenfilter stuff
```

```matlab
testeigfil3.m
% test the eigenfilter stuff
```

```matlab
testexlms.m
% Test the lms in a system identification setting
% Assume a Gaussian input
```

```matlab
testlms.m
% test the lms in an equalizer setting
% Assume a binary +/- 1 input.
```

```matlab
testmusic.m
% Test the music algorithm
```

```matlab
testnn1.m
% test the neural network stuff
```

```matlab
testnn2.m
% test the neural network stuff
% (run testnn1.m first to get the network trained)
%
% does some plots after the initial training is finished
```

```matlab
testnn3.m
% test the neural network stuff
% try different values of mu and alpha
% run testnn1 first to get the training data
```

```matlab
testrls.m
% test the rls in an equalizer setting
% Assume a binary +/- 1 input.
```

```matlab
testrls2.m
% test the rls in a system identification setting
% Assume a binary +/- 1 input.
```

```matlab
testrls2ex.m
% test the rls in a system identification setting
% Assume a binary +/- 1 input.
```

```matlab
testrlsex.m
% test the rls in an equalizer setting
% Assume a binary +/- 1 input.
```

```matlab
testrot.m
% test the procrustes rotation
```

```matlab
testtls.m
% Test tls stuff
```

```matlab
vq1.m
% Generate random Gaussian data, determine a codebook for it, and plot
```

```matlab
wftestcont.m
% make the contour plot for wftest
```
