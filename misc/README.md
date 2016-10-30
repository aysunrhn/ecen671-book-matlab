Directory: misc
===============

This is a treasure trove of functions which were written more or less in parallel with the book. Some categories of functions:
- number theoretical methods: including fast convolution, CRT for integers and polynomials, Euclidean algorithm, polynomial operations, linear congruences, and many others.
- interpolation and approximation: including Bernstein polynomials, Neville's algorithm, splines, rational approximation, remez (minimax), Vandermonde matrix equation solver, trigonometric interpolation, pade approximaation, etc.
- examples related to the book: including neural networks, Viterbi, Pisarenko, Prony, composite mappings, HMM training (with unscaled forward/backward probabilities), etc.

The source of many of these is a part of the book --- which was removed
due to lack of space --- about-number theoretical methods in signal
processing, and another part --- also removed --- on interpolation
and approximation.

-------------------------------------------------------

### TOOLBOXES:
The code written should run with Matlab without additional toolboxes,
with the following known exceptions:

`misc/kaisfilt.m` requires the use of the `sinc` and the `freqz` function (found in the signal processing toolbox)

`misc/ratintfilt.m` makes use of the `zplane` function for plotting (found in the signal processing toolbox)

--------------------------------------------------------

### Files:

```matlab
H.m
% 
% Compute the binary entropy function
% 
% function h = H(p)
%
% p = crossover probability
%
% h = binary entropy
```

```matlab
b2n.m
% convert an m-bit binary sequence b to an integer
```

```matlab
bernapprox.m
% Plot Bernstein polynomials
```

```matlab
bernpoly.m
% 
% compute the Bernstein polynomial g_{nk}(t)
%
% function g = bernpoly(n,k,t)
%
% n = degree
% k = order
% t = location
% 
% g = value
```

```matlab
bsplineval1.m
% 
% When f(x) = sum_i c_i^k B_i^k(x)
% where B_i^k is the kth order spline across t_i,t_{i+1},\ldots,t_{i+k+1}
% This function evaluates f(x) for a given x.
% (See Kincaid-Cheney, p. 396)
%
% function s = bsplineval1(c,t,x,k)
%
% c = set of coefficients
% t = knot set
% x = value.  x must fall in range of knots, t_i <= x < t_{i+1}
% k = order
%
% s = spline value
```

```matlab
chebinterp.m
% data for  Chebyshev interpolation example
```

```matlab
chi2.m
% 
% Compute the pdf for a chi-squared random variable
%
% function f = chi2(x,n)
%
% x = value
% n = order of chi-squared
%
% f = pdf value
```

```matlab
compmap1.m
% Test the composite mapping algorithm on the positive sequence
```

```matlab
compmap4.m
% test the positive semi-definite mapping
```

```matlab
computeremez.m
% Test the remez algorithm
```

```matlab
crtgamma.m
% 
% Compute the gammas for the CRT
%
% function gamma = crtgamma(m)
%
% m = set of modulos
%
% gamm = set of gammas
```

```matlab
crtgammapoly.m
% 
% Compute the gammas for the CRT for polynomials
%
% function gamma = crtgammapoly(m)
%
% m = set of modulos (polynomials)
%
% gamma = set of gammas (polynomials)
```

```matlab
crypttest.m
% test the cryptographic example
```

```matlab
d2b.m
% convert n to an m-bit binary representation
```

```matlab
diagstack.m
% Stack matrices diagonally:
% D = [X 0
%      0 Y];
%
% function D = diagstack(X,Y)
% X, Y = input matrices
%
% D = diagonal stack
```

```matlab
discapprox.m
% find a discrete approximating polynomial
```

```matlab
divdiff.m
% 
% Compute the upper row of a divided difference table
%
% function c = divdiff(ts,fs)
%
% ts = sample locations
% fs = function values
%
% c = set of divided differences
```

```matlab
eigfilcon0.m
%
% Find eigenilter constrained so that response is 0 at some frequencies
%
% function h = eigfilcon(wp,ws,N,alpha, Wo)
%
% wp = passband frequency
% ws = stopband frequency
% m = number of coefficients (must be ODD)
% alpha = tradeoff parameter between stopband and passband
% Wo = list of constraint frequencies at which response is 0
%
% h = filter coefficients
```

```matlab
eigfilcon0new.m
%
% Find eigenilter constrained so that response is 0 at some frequencies
%
% function h = eigfilcon(wp,ws,N,alpha, Wo)
%
% wp = passband frequency
% ws = stopband frequency
% m = number of coefficients (must be ODD)
% alpha = tradeoff parameter between stopband and passband
% Wo = list of constraint frequencies at which response is 0
%
% h = filter coefficients
```

```matlab
elem.m
%
% Return an elementary matrix E_{rs} of size mxn
```

```matlab
fact.m
% compute the factorial
```

```matlab
findprim5.m
% Find a primitive polynomial in GF(5)
```

```matlab
fromcrt.m
% 
% Given a sequence [y1,y2,\ldots,y2] that is a representation 
% of an integer x in CRT form, convert back to x.
%
% function x = fromcrt(y,m)
%
% y = sequence
% m = [m1,m2,...,mr]
% gammain (optional) = set of gamma factors.  
%   If not passed in, gamma is computed
%
% x = integer representation
% gamma = gamma values
```

```matlab
fromcrtpoly.m
% 
% Compute the representation of the polynomial f using the Chinese Remainder
% Theorem (CRT) using the moduli m = [m1,m2,...,mr].  It is assumed 
% (without checking) that the moduli are relatively prime.  
% Optionally, the gammas may be passed in (speeding computation), and
% are returned as optional return values.
%
% function [f,gamma] = fromcrtpoly(y,m,gammain)
% function [f] = fromcrtpoly(y,m)
% function [f,gamma] = fromcrtpoly(y,m)
% function [f] = fromcrtpoly(y,m,gammain)
%
% y = list of polynomials (cell array)
% m = list of moduli (cell array)
% gammain = (optional)list of gammas (cell array)
%
% f = reconstructed polynomial
% gamma = gamma
```

```matlab
fromhankel.m
%
% Pull sequential data out of a Hankel matrix X 
%
% X = Hankel matrix
% d = (optional) dimension of data
%
% x = Sequential data
```

```matlab
fromhankel2.m
% Pull sequential data out of a Hankel matrix X  (cell array version)
%
% X = Hankel matrix
% d = (optional) dimension of data
%
% x = Sequential data (in a cell array)
```

```matlab
gam1.m
% 
% Determine the optimum bet on a subfair track
%
% function [B,b0,b,l] = gam1(p,o)
%
% p = probability of win
% o = subfair odds
%
% B = other bets
% b0 = amount witheld
% b = bet
```

```matlab
gcdint1.m
% 
% Compute (only) the GCD (a,b) using the Euclidean algorithm
%
% function g = gcdint1(b,c)
%
% b,c = integers
%
% g = GCD(b,c)
```

```matlab
gcdint2.m
% 
% Compute the GCD g = (b,c) using the Euclidean algorithm
% and return s,t such that bs+ct = g
%
% function [g,s,t] = gcdint2(b,c)
%
% b,c = integers
% g = GCD(b,c)
% s,t = integers
```

```matlab
gcdpoly.m
% 
% Compute the GCD g = (b,c) using the Euclidean algorithm
% and return s,t such that bs+ct = g, where b and c are polynomials
% with real coefficients
%
% function [g,s,t] = gcdpoly(b,c)
%
% b,c = polynomials
% thresh = (optional) threhold argument used to truncate small remainders
%
% g = GCD(b,c)
% s,t = polynomials
```

```matlab
genardat.m
% 
% Generate N points of AR data with a = [a(1) a(2), \ldots, a(n)]'
% and input variance sigma^2
%
% function x = genardat(a,sigma,N)
%
% a = AR parameters
% sigma = standard deviation of input noise
% N = number of points
%
% x = AR process
```

```matlab
greedyperm.m
% 
% Using a greedy algorithm, determine a permutation P such that Px=z
% as closely as possible.
%
% function P = greedyperm(x,z)
```

```matlab
greedyperm2.m
% 
% Using a greedy algorithm, determine a permutation P such that Px=z
% as closely as possible.
% This algorithm is more complex than greedyperm
%
% function P = greedyperm2(x,z)
```

```matlab
haar1.m
% Do the computations for the Haar transform, working toward the
% wavelet lifting transform
%
% s = input data
% nlevel = number of levels
%
% h = Haar transform
```

```matlab
haarinv1.m
% Do the computations for the inverse Haar transform, working toward the
% wavelet lifting transform
```

```matlab
hmmApiup.m
% 
% Update the A and pi probabilities in the HMM using the forward and
% backward probabilities alpha and beta
%
% function [A,pi] = hmmapiup(y,alpha,beta,HMM)
%
% y = input sequence
% alpha = forward probability
% beta = backward probability
% f = distribution
% HMM = model parameters
%
% A = updated state transition probability
% pi = updated initial state probability
```

```matlab
hmmab.m
% 
% Compute the forward and backward probabilities for the model HMM
% and the output probabilities
```

```matlab
hmmdiscfup.m
% 
% Update the output probability distribution f of the HMM using the forward
% and backward probabilities alpha and beta
%
% function f = hmmdiscfup(y,alpha,beta,HMM)
%
% y = input sequence
% alpha = forward probabilities
% beta = backward probabilities
% HMM = current model parameters
%
% f = updated distribution
```

```matlab
hmmfupdate.m
% 
% Provide an update to the state output distributions for the HMM model
%
% function f = hmmfupdate(y,alpha,beta,HMM)
% 
% y = input sequence
% alpha = forward probabilities
% beta = backward probabilities
% HMM = current model parameters
% 
% f = updated distribution
```

```matlab
hmmgausfup.m
% 
% Update the Gaussian output distribution f of the HMM using the
% probabilities alpha and beta
%
% function f = hmmgausfup(y,alpha,beta,HMM)
% 
% y = input sequence
% alpha = forward probabilities
% beta = backward probabilities
% HMM = current model parameters
% 
% f = updated distribution
```

```matlab
hmmlpyseq.m
%
% Find the log likelihood of the sequence y[1],y[2],...,y[T], 
% i.e., compute log P(y|HMM)
%
% function lpy = hmmlpyseq(y,HMM)
%
% y = input sequence
% HMM = current model parameters
```

```matlab
hmmupdate.m
% 
% Compute updated HMM model from observations (nonnormalized)
%
% function HMM = hmmupdate(y,HMM) 
%
% y = output sequence
% HMM = current model parameters
%
% hmmo = updated model
```

```matlab
ifs2.m
% test some ifs stuff
```

```matlab
ifs2ex.m
% find an affine transformation Ax + b that transforms from
% {x00,x10,x20,x30} to {x01,x11,x21,x31}
```

```matlab
initvit2.m
% 
% Initialize the data structures and pointers for the Viterbi algorithm
%
% function initvit2(intrellis, inbranchweight, inpathlen, innormfunc)
%
% intrellis: a description of the successor nodes using a list, e.g.
%          intrellis{1} = [1 3]; intrellis{2} = [3 4];
%          intrellis{3} = [1 2]; intrellis{4} = [3 4];
% inbranchweight: weights of branches used for comparison, saved as
%    cells in branchweight{state_number, branch_number}
%    branchweight may be a vector
%    e.g.  branchweight{1,1} = 0; branchweight{1,2} = 6;
%          branchweight{2,1} = 3; branchweight{2,2} = 3;
%          branchweight{3,1} = 6; branchweight{3,2} = 0;
%          branchweight{4,1} = 3; branchweight{4,2} = 3;
% inpathlen: length of window over which to compute
% normfun: the norm function used to compute the branch cost
```

```matlab
interplane.m
% 
% find the intersecting point for the planes
%  m1'(x-x1) = 0   and  m2'(x-x2)=0.
%
% function x = interplane(m1,x1,m2,x2)
%
% m1 = normal to plane
% x1 = point on plane
% m2 = normal to plane
% x2 = point on plane
%
% x = intersecting point
```

```matlab
invdiff.m
% 
% Compute the inverse differences for a rational interpolation function,
% returning the vector of inverse differences that are necessary for
% interpolation
%
% function phis = invdiff(ts,fs)
%
% ts = vector of independent variable
% fs = vector of dependent variable
%
% phis = inverse differences
```

```matlab
jordanex.m
% example of Jordan forms
```

```matlab
kaisfilt.m
% test the design of a Kaiser filter
```

```matlab
kronsum.m
%
% Kronecker sum of A and B.  A and B are assumed square.
%
% function C = kronsum(A,B).  
```

```matlab
lagrangepoly.m
% 
% Lagrange interpolator
```

```matlab
latexform.m
% 
% Display a matrix X in latex form
%
%function latexform(fid,X,[nohfill])
% 
% fid = output file id (use 1 for terminal display)
% X = matrix of vector to display
% nohfill = 1 if no hfill is wanted
```

```matlab
lfsr.m
% 
% Produce m outputs of an lfsr with coefficient c and initial values y0
%
% function y = lfsr(c,y0,m)
% y0 = [y_0,...,y_{p-2},y_{p-1}]
% c = [c(1),c(2),...,c(p)]
% 
% y_j = sum_{i=1}^p y_{j-i} c(i)
```

```matlab
lfsrfind.m
% 
% Find a good lfsr c to match Ac=b
%
% function c = lfsrfind(A,b)
```

```matlab
lfsrfind2.m
% 
% Find a good lfsr c to match Ac=b
% where A and b are formed by the lfsr
% In this case, feed the error back around
%
% function c = lfsrfind2(y,m)
```

```matlab
lsdata.m
% Make least-squares data matrices 
```

```matlab
makemarkov.m
% 
% Return the sequence of n impulse response samples into the cell array y
% y{1},y{2}, ... y{n}
%
% function y = makemarkov(A,B,C,n)
%
% (A,B,C) = system
% n = number of samples
%
% y = cell array of impulse responses
```

```matlab
makeperm.m
% Return all permutations of length n
```

```matlab
maketoeplitz.m
% 
% Form a toeplitz matrix from the input data y
%
% function [H] = maketoeplitz(y,m,n)
%
% y = input data = [y1 y2 ...] (a series of vectors in a _row_)
% m = number of block rows in H
% n = number of block columns in H
```

```matlab
marv.m
% 
% Prony: given a sequence of (supposedly) pure sinusoidal data, 
% determine the frequency using model methods
%
% function f = marv(x,fs)
% x = data sequence
% fs = sampling rate
%
% f = frequencies found
```

```matlab
masseyinit.m
% Initialize the iteratively called massey's algorithm
%
% function masseyinit()
```

```matlab
masseyit.m
% 
% Compute the lfsr connection polynomial using Massey's algorithm
% accepting new data at each iteration.
% masseyinit should be called before calling this the first time
%
% y = new data point
%
% c = updated connection polynomial
```

```matlab
miniapprox1.m
% minimax approximation example
```

```matlab
modaldata1.m
% data for a modal analysis problem
```

```matlab
myplaysnd.m
% 
% Modified and simplified from playsnd, to make the sample rate stuff work
```

```matlab
n2b.m
% convert n to an m-bit binary representation
```

```matlab
neville.m
% 
% Neville's algorithm for computing a value for an interpolating polynomial
% y = NEVILLE(x,X,Y) takes the (xi,yi) coordinate pairs in the
% vectors X and Y and computes the value of the unique
% interpolating polynomial that passes through the list of points
% at the given value of x. 
% 
% function y = neville(x, X, Y)
%
% x = interpolated point
% X = X data
% Y = Y data
%
% y = interpolated value
```

```matlab
nntrain2.m
% 
% train a neural network using the input/output training data [x,d]
% with sequential selection of the data
%
% function w = nntrain(x,d,m,ninput,mu)
%
% x = [x(1) x(2) ... x(N)]   
% d= [d(1) d(2) ... d(N)]
% nlayer = number of layers
% m = number of neurons on each layer, 
%     m(1) = input layer, ... m(nlayer+1) = ouput layer
% mu = steepest descent step size
% alpha = (optional) momentum constant
% maxiter = (optional) maximum number of iterations (w = no maximum)
% w = (optional) starting weights
%
% err = (optional) total squared error from training
```

```matlab
pade1.m
% Pade example
```

```matlab
padefunct.m
% 
% Find the Pade approximation from the Maclaurin series coefficients
%
% function [A,B] = padefunct(c,m,k)
%
% c = Maclaurin series coefficients (need m+k+1)
% m = degree of numerator polynomial
% k = degree of denominator polynomial
%
% A = coefficients of numerator polynomial (in Matlab order)
% B = coefficients of denominator polynomial (in Matlab order)
```

```matlab
permer.m
% 
% function permlist = permer(n1,p,perm,permnew,permlist)
```

```matlab
pisexamp.m
% Example for Pisarenko Harmonic Decomposition
```

```matlab
plotbernapprox.m
% plot the Benstein polynomial approximation to $f(t) = e^t$
```

```matlab
plotbernpoly.m
% plot the Benstein polynomial
```

```matlab
plotfplane.m
% plot a function and a linear approximating surface
```

```matlab
plotplane.m
% determine points in the plane m'(x-x0) = 0 for plotting purposes
% 
```

```matlab
polyadd.m
%
% Add the polynomials p=a+b
% 
% function p = polyadd(a,b)
%
% a,b = polynomial
%
% p = polynomial sum.
```

```matlab
polydiv.m
% 
% Divide a(x)/b(x), and return quotient and remainder in q and r
% Coefficients are assumed to be in Matlab standard order (highest order first)
%
%
% function [q,r] = polydiv(a,b)
%
% a = numerator
% b = denominator
%
% q = quotient
% r = remainder
```

```matlab
polydivgfp.m
% 
% Divide a(x)/b(x), and return quotient and remainder in q and r
% using arithmetic in GF(p)
% Coefficients are assumed to be in Matlab standard order (highest order first)
%
% function [q,r] = polydivgfp(a,b)
%
% a = numerator
% b = denominator
%
% q = quotient
% r = remainder
```

```matlab
polymult.m
% 
% Multipoly the polynomials p=a*b
%
% function p = polymult(a,b)
%
% a,b = polynomials
%
% p = product
```

```matlab
polysub.m
% 
% Subtract the polynomials p=a-b
%
% a,b = polynomials
%
% p = difference
```

```matlab
psdarma.m
%
% Plot the psd of an arma model
%
% function [w,h] = psdarma(b,a)
%
% b = numerator coefficients
% a = denominator coefficients
%
% w = frequency values
% h = absolute value of response
```

```matlab
ratinterp.m
% 
% Compute the rational function interpolation
% from the data in ts and fs.
% Polynomial coefficients returned in Matlab order (largest to smallest)
```

```matlab
ratinterp1.m
% 
% Compute a single interpolated point f(t) given the interpolating data
% and the inverse differences
%
% function f = ratinterp1(t,ts,fs,phis)
%
% t = point at which to evaluate
% ts = vector of independent data
% fs = vector of depdendent data
% phis = inverse differences
%
% f = interpolated value
```

```matlab
ratintfilt.m
% Try some data for a rationally-interpolated filter
```

```matlab
res.m
% 
% Computes <a^n>_m
%
% function d = res(a,n,m)
%
% a = value
% n = exponent
% m = modulo
%
% d = remainder(a^n,m0
```

```matlab
schurcohn.m
% 
% Returns 1 if p is a Schur polynomial (all roots inside unit circle)
%
% function stable = schurcohn(p)
% 
% p = polynomial coefficients
%
% stable = 1 if stable polynomial
```

```matlab
simppivot.m
% 
% Pivot a linear programming tableau about the p,q entry
% 
% function tableau = simppivot(intableau,p,q)
%
% intableau = tableau
% (p,q) = point about which to pivot
%
% tableau = pivoted tableau
```

```matlab
solvlincong.m
% 
% Ddetermine the solution to the linear congruence
% a x equiv b (mod m), if it exists
%
% function x = solvlincong(a,m,b)
```

```matlab
sreal.m
% sysreal.m
% data for the system identification example in the SVD stuff
```

```matlab
sreal1.m
% SVD realization
```

```matlab
sugiyama.m
% 
% Compute the GCD g = (b,c) using the Euclidean algorithm
% and return s,t such that bs+ct = g, where b and c are polynomials
% with real coefficients
%
% thresh = (optional) threshold argument used to truncate small remainders
```

```matlab
sysidsvd2.m
% 
% given a sequence of impulse responses in h (a cell array)
% identify a system (A,B,C)
% This uses the tohankel method of finding a nearest hankel matrix
% of desired rank
%
% function [A,B,C] = sysidsvd(h,order)
%
% h = impulse response sequence (cell array)
% order = desired order of system
%
% (A,B,C) = system
```

```matlab
taylorf.mm
(* example of a taylor series *)
```

```matlab
tocrt.m
% 
% Compute the representation of the scalar x using the
% using the Chinese Remainder Theorem (CRT) with
% moduli m = [m1,m2,...,mr].  It is assumed (without checking)
% that the moduli are relatively prime
%
% function y = tocrt(x,m)
%
% x = number to convert
% m = set of moduli
%
% y = CRT representation of x
```

```matlab
tocrtpoly.m
% 
% Compute the representation of the polynomial f using the
% using the Chinese Remainder Theorem (CRT) with
% moduli m = {m1,m2,...,mr}.  It is assumed (without checking)
% that the moduli are relatively prime.
% m is passed in as a cell array containing polynomial vectors
% and y is returned as a cell array containing polynomial vectors
%
% function y = tocrt(f,m)
%
% f = polynomial
% m = set of modulo polynomials
%
% y = CRT form of f
```

```matlab
tohankelbig.m
% 
% Determine the matrix nearest to A which is (block) Hankel and has rank r
% using the composite mapping algorithm
%
% function A = tohankelbig(A,r)
%
% A = input matrix
% r = desired ranke
% d = (optional) block size (default=1)
%
% A = nearest rank r Hankel matrix
% diff = norm of difference between matrices
```

```matlab
triginterp.m
% demonstrate trigonometric interpolation
```

```matlab
vandsolve1.m
%
% Solves the equation Vx = fs, where V is the Vandermonde
% matrix determined from ts.
%
% function a = vandsolve1(ts,fs)
%
% ts = abscissa values
% fs = ordinate values
%
% a = solution
```

```matlab
vitnop.m
% 
% Compute the norm of the difference between inputs
% This function may be feval'ed for use with the Viterbi algorithm
% In this case, the norm is simply taken as the branch number
%
% function d = vitnop(branch,input)
%
```

```matlab
vitsqnorm.m
% 
% Compute the square norm of the difference between inputs
% This function may be feval'ed for use with the Viterbi algorithm
% (state and nextstate are not used here)
%
% function d = vitsqnorm(branch,input,state,nextstate)
```

```matlab
wino3by3.m
% 
% Convolve the 3-sequence a with the 3-sequence b 
% a and b are both assumed to be column vectors
% using Winograd convolution
%
% function c = wino3by3(a,b) 
```

```matlab
winotest.m
% Set up data for a Winograd convolution algorithm
```

```matlab
winotest2.m
% Set up data for a Winograd convolution algorithm 
```
