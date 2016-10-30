Directory: bookalgs
===================

### TOOLBOXES:
The code written should run with Matlab without additional toolboxes,
with the following known exceptions:

`bookalgs/eigmakePQ.m` requires the use of the `sinc` function (found in the signal processing toolbox)

`bookalgs/testirwls.m` makes use of the `freqz` function for plotting (found in the signal processing toolbox)

------------------------------------------------------

### Files: ###

```matlab
art1.m
% 
% Produce an updated solution x to Ax = b using the algebraic reconstruction
% technique.
% (This function is for demonstration, and does not precompute
% row norms as a more efficient version would.)  No constraints are applied.
%
% function x = art1(A,x,b)
%
% A = input matrix
% x = initial solution
% b = right-hand side
% 
% Output x= updated solution
```

```matlab
bayes3.m
% Minimax decision for Gaussian
```

```matlab
bidiag.m
% 
% bidiagonalize the real symmetric matrix A: UAV' = B,
% 
% function [B,U,V] = bidiag(A)
%
% A = matrix to be factored
%
% B = bidiagonal
% U = orthogonal transformation (optional output)
% V = orthogonal transformation (optional output)
```

```matlab
bliter1.m
% Iterate on bandlimited data using two projections
```

```matlab
bss1.m
% Blind source separation example
```

```matlab
bssg.m
%
% The nonlinear function in the neural network
%
% function y = bssg(v).  
```

```matlab
cholesky.m
%
% Compute the Cholesky factorization of B, B = LL'
% (this version does not require additional storage)
%
% function [L] = cholesky(B)
%
% B = matrix to be factored
%
% L = lower triangular factor
```

```matlab
compmap2.m
% Run the composite mapping algorithm to make a positive sequence
```

```matlab
conjgrad1.m
% function [x,D] = conjgrad1(Q,b)
%
% Solve the equation Qx = b using conjugate gradient, where Q is symmetric
%
% Q = symmetric matrix
% b = right-hand side
```

```matlab
conjgrad2.m
%
% Apply the conjugate gradient to minimize a function
%
% function [x,X] = conjgrad2(x,grad,hess)
%
% x = starting point
% grad = the gradient of the function to minimize (a function name)
% hess = the gradient of the function to minimize (a function name)
%
% Output x = update of point
% X = array of points examined (optional)
```

```matlab
convnorm.m
%
% Compute the Hamming distance between the branchweights and the input
% This function may be feval'ed for use with the Viterbi algorithm
% (state and nextstate are not used here)
% 
% function d = convnorm(branch,input,state,nextstate)
% 
```

```matlab
dijkstra.m
% 
% Find the set of shortest paths from vertex a to each of the other vertices
% in the graph represented by the cost matrix.
% If there is no branch from i to j, then cost(i,j) = inf.
%
% function [dist,prevnode] = dijkstra(a,cost)
% 
% a = starting vectex
% cost = cost matrix for graph
%
% dist(i) = shortest distance from vertex a to vertex i.
% prevnode(i) = vertex prior to vertex i on the shortest path to i.
```

```matlab
dirtorefl.m
% 
% Convert from direct-form FIR filter coefficients in a
% to lattice form
%
% function kappa = refltodir(a)
%
% a = direct form coefficients
%
% kapp = lattice form coefficients
```

```matlab
durbin.m
% 
% solve the nxn Toeplitz system Tx = [r(1)..r(n)]
% given a vector r = (r_0,r_1,\ldots,r_{n+1}),
%
% Since matlab has no zero-based indexing, r(1) = r_0
%
% function [x] = durbin(r)
%
% r = input vector
% x = solution to Tx = r
```

```matlab
eigfil.m
%
% Design an eigenfilter
% 
% function h = eigfil(wp,ws,m,alpha)
%
% wp = passband frequency
% ws = stopband frequency
% m = number of coefficients (must be ODD)
% alpha = tradeoff parameter between stopband and passband
%
% h = filter coefficients
```

```matlab
eigfilcon.m
%
% Design an eigenfilter with values constrained at some frequencies
%
% function h = eigfilcon(wp,ws,N,alpha, Wo, d)
%
% wp = passband frequency
% ws = stopband frequency
% m = number of coefficients (must be ODD)
% alpha = tradeoff parameter between stopband and passband
% Wo = list of constraint frequencies
% d = desired magnitude values at those frequencies
%
% h = filter coefficients
```

```matlab
eigmakePQ.m
%
% Make the P and Q matrices for eigenfiltering
%
% function [P,Q] = eigmakePQ(wp,ws,N)
%
% wp = passband frequency
% ws = stopband frequency
% N = number of coefficients
```

```matlab
eigqrshiftstep.m
% 
% Perform the implicit QR shift on T, where the shift
% is obtained by the eigenvalue of the lower-right 2x2 submatrix of T
%
% function [T,Q] = eigqrshiftstep(T,Qin)
% 
% T = tridiagonal matrix
% Qin = (optional) orthogonal input matrix
%
% T = T matrix after QR shift
% Q = (optional) orthogonal matrix
```

```matlab
em1.m
% Illustration of example em algorithm computations
%
```

```matlab
esprit.m
%
% Compute the frequency parameters using the ESPRIT method
%
%function f = esprit(Ryy,Ryz,p)
%
%
% Ryy = (estimate of) the autocorrelation of the observations
% Ryz = (estimate of) the correlation between y[n] and y[n+1]
% p = number of modes
%
% f = vector of frequencies (in Hz/sample)
```

```matlab
et1.m
% et1.m
% Perform tomographic reconstruction using the EM algorithm
% im is a (nr x nc) image scaled so that 0=white, 255=black
% nr = number of rows;   nc = number of columns
```

```matlab
fblp.m
%
% Forward-backward linear predictor
%
% function [a,sigma] = fblp(x,n)
%
% Given a sequence of data in the columns vector x, 
% return the coefficients of an nth order forward-backward linear predictor in a
%
% x = data sequence
% n = length of filter
%
% a = filter coefficients
% sigma = standard deviation of noise
```

```matlab
fordyn.m
%
% Do forward dynamic programing
%
% function [pathlist,cost] = fordyn(G,W)
%
% G = list of next vertices
% W = list of costs to next vertices
%
% pathlist = list of paths through graphs
% cost = cost of the paths
```

```matlab
gamble.m
%
% Return the bets for a race with win probabilities p and subfair odds o,
%
% function [b0,B] = gamble(p,o)
%
% p = probability of win
% o = subfair odds
```

```matlab
gaussseid.m
% 
% Produce an updated solution x to Ax = b using Gauss-Seidel iteration
%
% function x = gausseid(A,x,b)
%
% A = input matrix
% x = initial solution
% b = right-hand side
% 
% Output x= updated solution
```

```matlab
golubkahanstep.m
% 
% Given a bidiagonal matrix B with NO zeros on the diagonal or
% superdiagonal, create a new B <-- U'BV, using an implicit QR shift on
% T = B'B
%
% function [B,U,V] = golubkahanstep(B,Uin,Vin)
%
% B = bidiagonal matrix
% Uin, Vin = last estimate of U and V
%
% B = new bidiagonal matrix
% U, V = new estimate of U and V
```

```matlab
gramschmidt1.m
% 
% Compute the Gram-Schmidt orthogonalization of the 
% columns of A, assuming nonzero columns of A
%
% function [Q,R] = gramschmidt1(A)
%
% A = input matrix to be factored (assuming nonzero columns)
%
% Q = orthogonal matrix
% R = upper triangular matrix such that A = QR
```

```matlab
hmmApiupn.m
% 
% update the HMM probabilities A and pi using the normalized forward and
% backward probabilities alphahat and betahat
%
% function [A,pi] = hmmapiupn(y,alphahat,betahat,HMM)
%
% y = input sequence
% alphahat = normalized alphas
% betahat = normalized betas
% HMM = current model parameters
%
% A = updated transition probability matrix
% pi = updated initial probability matrix
```

```matlab
hmmabn.m
% 
% compute the normalized forward and backward probabilities for the model HMM
% and the output probabilities and the normalization factor
%
% function [alphahat,betahat,f,c] = hmmabn(y,HMM)
%
% y = input sequence y[1],y[2], ..., y[T]
% HMM = HMM data
%
% alphahat = [ alphahat(:,1) alphahat(:,2) ... alphahat(:,T)]
%            (alphahat(:,t) = alphahat(y_t^T,:))
% betahat = [betahat(:,2) ... betahat(:,T-1) betahat(:,T) betahat(:,T+1)]
%            (betahat(:,t) = betahat(y_{t+1}^T|:))
% f = initial probability types
% c = normalizing factors
```

```matlab
hmmdiscf.m
%
% Compute the pmf value for a discrete distribution
%
% function f = hmmdiscf(f)
% 
% y = output value
% f = output distribution
% s = state
```

```matlab
hmmdiscfupn.m
% 
% Update the discrete HMM output distribution f using the normalized forward
% and backward probabilities alphahat and betahat
% 
% function f = hmmdiscfupn(y,alphahat,betahat,c,HMM)
%
% y = output sequence
% alphahat = normalized alphas
% betahat = normalized betas
% c = normalization constants
% HMM = current model parameters
%
% f = updated output distribution
```

```matlab
hmmf.m
% 
% Determine the likelihood of the output y for the model HMM
% This function acts as a clearinghouse for different probability types
%
% function p = hmmf(y,f,s)
%
% y = output value
% f = probability distribution
% s = state
%
% p = likelihood
```

```matlab
hmmfupdaten.m
% 
% Provide an update to the state output distributions for the HMM model
% using the normalized probabilities alphahat and betahat
%
% function f = hmmfupdaten(y,alphahat,betahat,HMM)
%
% y = output sequence
% alphahat = normalized alphas
% betahat = normalized betas
% c = normalization factors
% HMM = current model paramters
%
% f = updated output distribution
```

```matlab
hmmgausf.m
% 
% Compute the pmf value for a Gaussian distribution
%
% function f = hmmgausf(y,f,s)
%
% y = output value
% f = output distribution
% s = state
```

```matlab
hmmgausfupn.m
% 
% Update the Gaussian output distribution f of the HMM using the normalized
% probabilities alphahat and betahat
%
% function f = hmmgausfupn(y,alphahat,betahat,c,HMM)
%
% y = output sequence
% alphahat = normalized alphas
% betahat = normalized betas
% c = normalization constants
% HMM = current model parameters
%
% f = updated output distribution
```

```matlab
hmmgendat.m
% 
% Generate T outputs of a Hidden Markov model HMM
%
% function [y,ss] = hmmgendat(T,HMM)
%
% T = number of samples to produce
% HMM = HMM model parameters
%
% y = output sequence
% s = (optional) state sequence
```

```matlab
hmmgendisc.m
% 
% Generate T outputs of a HMM with a discrete output distribution
%
% function y = hmmgendisc(T,HMM)
% 
% T = number of outputs to produce
% HMM = model parameters
%
% y = output sequence
% ss = (optional) state sequence (for testing purposes)
```

```matlab
hmmgengaus.m
% 
% Generate T outputs of a HMM with a Gaussian output distribution
%
% function y = hmmgengaus(T,HMM)
%
% T = number of outputs to produce
% HMM = model parameters
%
% y = output sequence
% ss = (optional) state sequence (for testing purposes)
```

```matlab
hmminitvit.m
% 
% Initialize the Viterbi algorithm stuff for HMM sequence identification
%
% function hmminitvit(inHMM,inpathlen)
% 
% inHMM = a structure containing the initial probabilities, state transition
%         probabilities, and output probabilities
% inpathlen = length of window used in VA
```

```matlab
hmmlpyseqn.m
%
% Find the log likelihood of the sequence y[1],y[2],...,y[T],
% using the parameters in HMM.% That is, compute log P(y|HMM),
%
% function lpy = hmmlpyseqn(y,HMM)
%
% y = input sequence = y[1],y[2],\ldots,y[T]
% HMM = current estimate of HMM parameters
%
% lpy = log P(y|HMM)
```

```matlab
hmmlpyseqv.m
% 
% Find the log likelihood of the sequence y[1],y[2],...,y[T]
% i.e., compute log P(y|HMM),
% using the the Viterbi algorithm
%
% function lpy = hmmlpyseqv(y,HMM)
%
% y = input sequence
% HMM = HMM parameters
%
% lpy = log likelihood value
```

```matlab
hmmnorm.m
% 
% Compute the branch norm for the HMM using the Viterbi approach
%
% function d = hmmnorm(branchweight,y,state,nextstate)
%
% branchweight= log transition probability 
% y = output
% state = current state in trellis
% nextstate = next state in trellis
%
% d = branch norm (log-likelihood)
```

```matlab
hmmnotes.m
% Notes on data structures and functions for the HMM
% 
```

```matlab
hmmtest2vb.m
% Test the HMM using both Viterbi and EM-algorithm based training methods
```

```matlab
hmmupdaten.m
%
% Compute updated HMM model from observations
%
% function HMM = hmmupdaten(y,HMM) 
%
% y = output sequence
% HMM = current model parameters
%
% hmmo = updated model parameters
```

```matlab
hmmupdatev.m
% 
% Compute updated HMM model from observations y using Viterbi methods
% Assumes only a single observation sequence.
%
% function HMM = hmmupdatev(y,HMM) 
%
% y = sequence of observations
% HMM = old HMM (to be updated)
%
% hmmo = updated HMM
```

```matlab
hmmupfv.m
% 
% Compute an update to the distribution f based upon the data y
% and the (assumed) state assignment in statelist
%
% function fnew = hmmupfv(y,statelist,n,f)
%
% y = sequence of observations
% statelist = state assignments
% n = number of states
% f = distribution (cell) to update
%
% fnew = updated distribution
```

```matlab
houseleft.m
%
% Apply the Householder transformation based on v to A on the left
%
% function A = houseleft(A,v)
%
% A = an mxn matrix
% v = a household vector
%
% B = H_v A
```

```matlab
houseright.m
%
% Apply the householder transformation based on v to A on the right
%
% function A = houseright(A,v)
%
% A = an mxn matrix
% v = a household vector
%
% B = H_v A
```

```matlab
ifs3a.m
% Plot the logistic map and the orbit of a point
%
```

```matlab
initcluster.m
%
% 
% Choose an initial cluster at random
% 
% function Y = initcluster(X,m)
%
% X = input data: each column is a training data vector
% m = number of clusters
% Y = initial cluster: each column is a point
```

```matlab
initpoisson.m
% 
% Initialize the global variables for the poisson generator
% 
% function initpoisson
```

```matlab
initvit1.m
% 
% Initialize the data structures and pointers for the Viterbi algorithm
% 
% function initvit1(intrellis, inbranchweight, inpathlen, innormfunc)
%
% intrellis: a description of the successor nodes 
%    e.g. [1 3; 3 4; 1 2; 3 4]
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
invwavetrans.m
%
% Compute the inverse discrete wavelet transform 
%
% function c = invwavetrans(C,ap,coeff)
%
% C = input data (whose inverse transform is to be found)
% ap = index for start of coefficients for the jth level
% coeff = wavelet coefficients
%
% c = inverse transformed data
```

```matlab
invwavetransper.m
%
% Compute the periodized inverse discrete wavelet transform 
%
% function c = invwavetransper(C,coeff,J)
%
% C = input data
% coeff = wavelet coefficients
% J = (optional) number of levels of inverse transform to compute
%    If length(C) is not a power of 2, J must be specified.
%
% c = inverse discrete wavelet transform of C
```

```matlab
irwls.m
% 
% Computes the minimum solution c to ||x-Ac||_p using
% iteratively reweighted least squares
%
% function c = irwls(A,x)
%
% A = system matrix
% x = rhs of equation
% p = L_p norm
%
% c = solution vector
```

```matlab
jacobi.m
% 
% Produce an updated solution x to Ax = b using Jacobi iteration
%
% function x = jacobi(A,x,b)
%
% A = input matrix
% x = initial solution
% b = right-hand side
% 
% Output x= updated solution
```

```matlab
kalex1.m
% Kalman filter example 1
%
```

```matlab
kalman1.m
% 
% Computes the Kalman filter esimate xhat(t+1|t+1)
% for the system x(t+1) = Ax(t) + w
%                y(t) = Cx(t) + v
% where cov(w) = Q  and cov(v) = R, 
% The prior estimate is x0, and the prior covariance is P0.
% 
```

```matlab
karf.m
%
% Evaluate the potential function f(x,c)
% for karmarkers algorithm
%
% function f = karf(x,c)
% 
% x = value of x
% c = constraint vector
%
% f = potential function
```

```matlab
karmarker.m
% 
% Implement a Karmarker-type algorithm for linear programming
% to solve a problem in "Karmarker standard form"
%  min       c'x
% subject to Ax=0,  sum(x)=1, x >=0
%
% function x = karmarker(A,c)
%
% A,c = system matrices
%
% x = solution
```

```matlab
kissarma.m
% 
% Determine the ARMA parameters a and b of order p based upon the data in y.
%
% function [a,b] = kissarma(y,p)
%
% y = sequence 
% p = order of AR part
%
% a = AR coefficients
% b = MA coefficients
```

```matlab
levinson.m
% 
% Given a vector r = (r_0,r_1,\ldots,r_{n-1}),
% and a vector b = (b_1,b_2,\ldots,b_n)
% solve the nxn Toeplitz system Tx = b
%
% function [y] = levinson(r,b)
%
% Since Matlab has no zero-based indexing, r(1) = r_0
%
% r = vector of coefficients for Toeplitz matrix
% b = right-hand side
% 
% y = solution to Tx = b
```

```matlab
lgb.m
% 
% Find m clusters on the data X
%
% function [Y,d] = lgb(X,m)
%
%
% X = input data: each column is a training data vector
% m = number of clusters to find
%
% Y = set of clusters: each column is a cluster centroid
% d = minimum total distortion
```

```matlab
lms.m
% 
% Given a (real) scalar input signal x and a desired scalar signal d,
% compute an LMS update of the weight vector h.
% This function must be initialized by lmsinit
%
% function [h,eap] = lms(x,d)
%
% x = input signal (scalar)
% d = desired signal (scalar)
%
% h = updated LMS filter coefficient vector
% eap = (optional) a-priori error
```

```matlab
lmsinit.m
%
% Initialize the LMS filter
% 
% function lmsinit(m,mu)
%
% m = dimension of vector
% mu = lms stepsize
```

```matlab
logistic.m
% 
% Compute the logistic function y = lambda*x*(1-x)
%
% function y = logistic(x,lambda)
%
% x = input value (may be a vector)
% lambda = factor of the function
```

```matlab
lpfilt.m
% 
% Design an optimal linear-phase filter using linear programming
%
% function [h,delta] = lpfilt(fp,fs,n)
%
% fp = pass-band frequency  (0.5=Fs/2)
% fs = stop-band frequency
% n = number of coefficients (assumed odd here)
%
% h = filter coefficients
% delta1 = pass-band ripple
```

```matlab
lsfilt.m
%
% Determine a least-squares filter h with m coefficients 
%
% function [h,X] = lsfilt(f,d,m,type)
%
% f = input data
% d = desired output data
% m = order of filter
% type = data matrix type
%     type=1: "covariance" method    2: "autocorrelation" method
%          3: prewindowing           4: postwindowing
%
% h = least-squares filter
% X = (optional) data matrix
```

```matlab
makehankel.m
% 
% form a hankel matrix from the input data y
%
% [H] = makehankel(y,m,n)
%
% y = input data  = [y1 y2 ...] (a series of vectors in a _row_)
% m = number of block rows in H
% n = number of block columns in H
%
% H = Hankel matrix formed from y
```

```matlab
makehouse.m
%
% Make the Householder vector v such that Hx has zeros in 
% all but the first component
%
% function v = makehouse(x)
%
% x = vector to be transformed
%
% v = Householder vector
```

```matlab
massey.m
%
% Return the shortest binary (GF(2)) LFSR consistent with the data sequence y
%
% function [c] = massey(y)
%
% y = input sequence 
%
% c = LFSR connections, c = 1 + c(2)D + c(3)D^2 + ... c(L+1)D^L
%     (Note: opposite from usual Matlab order)
```

```matlab
maxeig.m
%
% Compute the largest eigenvalue and associated eigenvector of 
% a matrix A using the power method
%
% function [lambda,x] = maxeig(A)
%
% A = matrix whose eigenvalue is sought
%
% lambda = largest eigenvalue
% x = corresponding eigenvector
```

```matlab
mineig.m
% 
% Compute the smallest eigenvalue and associated eigenvector of 
% a matrix A using the power method
% function [lambda,x] = mineig(A)
%
% A = matrix whose eigenvalue is sought
%
% lambda = minimum eigenvalue
% x = corresponding eigenvector
```

```matlab
musicfun.m
%
% Compute the "MUSIC spectrum" at a frequency f.
%
% function pf = musicfunc(f,p,V)
%
% f = frequency (may be an array of frequencies)
% p = order of system
% V = eigenvectors of autocorrelation matrix
%
% pf = plotting value for spectrum
```

```matlab
neweig.m
% 
% Compute the eigenvalues and eigenvector of a real symmetric matrix A
%
% function [T,Q] = neweig(A)
%
% A = matrix whose eigendecomposition is sought
%
% T = diagonal matrix of eigenvalues
% Q = (optional) matrix of eigenvectors
```

```matlab
newlu.m
%
% Compute the lu factorization of A
%
% function [lu,indx] = newlu(A)
%
% A = matrix to be factored
%
% lu = matrix containg L and U factors
% indx = index of pivot permutations
```

```matlab
newsvd.m
% 
% Compute the singular value decomposition of the mxn matrix A, as A= u s v'.
% We assume here that m>n
% 
% [u,s,v] = newsvd(A)
% or
% s = newsvd(A)
%
% A = matrix to be factored
% 
% Output:
% s = singular values
% u,v = (optional) orthogonal matrices
```

```matlab
nn1.m
%
% Compute the output of a neural network with weights in w
%
% function [y,V,Y] = nn1(xn,w)
% 
% xn = input
% w = cell array of weights
%
% y = output layer output
% V = (optional) internal activity
% Y = (optional) neuron output
%    The optional arguments V and Y are used for training to store output for
%    each layer:
%    Y{1} = input, Y{2} = first hidden layer, etc.
%    V{1} = first hidden layer, etc.
```

```matlab
nnrandw.m
% 
% Generate an initial set of weights for a neural network at random,
% based upon the list in m
%
% function w = nnrandw(m)
% m = list of weights 
%   m(1) = number of inputs, m(2) = first hidden layer, etc.
%
% w = random weights
```

```matlab
nntrain1.m
%
% Train a neural network using the input/output training data [x,d]
%
% function w = nntrain(x,d,m,ninput,mu)
%
% x = [x(1) x(2) ... x(N)] = input training data   
% d = [d(1) d(2) ... d(N)] = output training data
% nlayer = number of layers
% m = number of neurons on each layer, 
%     m(1) = input layer, ... m(nlayer+1) = ouput layer
% mu = steepest descent step size
% alpha = (optional) momentum constant
% maxiter = (optional) maximum number of iterations (w = no maximum)
% w = (optional) starting weights
%
% w = new weights
% err = (optional) total squared error from training
```

```matlab
permutedata.m
% 
% Randomly permute the columns of the data x.
%
% function xp = permutedata(x,type)
%
% x = data to permute
% type=type of permutation
%   type=1: Choose a random starting point, and go sequentially
%   type=2: random selection without replacement (not really a permutation)
%
% xp = permuted x
```

```matlab
pisarenko.m
%
% Compute the the modal frequencies using Pisarenko's method,
% then find the amplitudes
%
% function [f,P] = pisarenko(Ryy)
%
% Ryy = autocorrelation function from observed data
%
% f = vector of frequencies
% P = vector of amplitudes
```

```matlab
pivottableau.m
% 
% Perform pivoting on an augmented tableau until 
% there are no negative entries on the last row
%
% function [tableau,basicptr] = pivottableau(intableau,inbasicptr)
%
% intableau = input tableau tableau,
% inbasicptr = a list of the basic variables, such as [1 3 4]
%
% tableau = pivoted tableau 
% basicptr = new list of basic variables
```

```matlab
poisson.m
% 
% Generate a sample of a random variable x with mean lambda
% (Following Numerical Recipes in C, 2nd ed., p. 294)
% This function should be initialized by initpoisson.m
%
% function x = poisson(lambda)
%
% lambda = Poisson mean
%
% x = Poisson random variable
```

```matlab
ptls1.m
% 
% Compute the Partial Total Least Squares solution of Ax = b
% where the first k columns of A are not modified
%
% function [x,Ahat,bhat] = ptls1(A,b,k)
%
% A = system matrix
% b = right-hand side
% k = number of columns of A not modified
%
% x = ptls solution to Ax=b
% Ahat = modified A matrix
% bhat = modified b matrix
```

```matlab
ptls2.m
% 
% Find the partial total least-squares solution to Ax = b,
% where k1 rows and k2 columns of A are unmodified
% 
% function [x] = ptls2(A,b,k1,k2)
%
% A = system matrix
% b = right-hand side
% k1 = number of rows of A not modified
% k2 = number of columns of A not modified
%
% x = PTLS solution to Ax=b
```

```matlab
qf.m
% 
% Compute the Q function:
%
% function p = qf(x)
%   p = 1/sqrt(2pi)int_x^infty exp(-t^2/2)dt
```

```matlab
qfinv.m
% 
% Compute the inverse of the q function
%
% function x = qfinv(q)
```

```matlab
qrgivens.m
% 
% Compute the QR factorization of a matrix A without column pivoting
% using Givens rotations
%
% function [R,thetac,thetas] = qrgivens(A)
% 
% A = mxn matrix (assumed to have full column rank)
%
% R = upper triangular matrix
% thetac = matrix of c values 
% thetas = matrix of s values (these must be converted to Q)
```

```matlab
qrhouse.m
% 
% Compute the QR factorization of a matrix A without column pivoting
% using Householder transformations
% 
% function [V,R] = qrhouse(A)
%
% 
% A = mxn matrix (assumed to have full column rank)
%
% V = [v1 v2 ... vn] = matrix of Householder vectors 
%           (must be converted to Q ) using qrmakeq.
%           
% R = upper triangular matrix
```

```matlab
qrmakeq.m
% 
% Convert the V matrix returned by qrhouse into a Q matrix
% 
% function Q = qrmakeq(V)
%
% V = [v1 v2 .... vr], Householder vectors produced by qrhouse
%
% Q = [Q1 Q2 ... Qr], the orthogonal matrix in the QR factorization
```

```matlab
qrmakeqgiv.m
% 
% Given thetac and thetas containing rotation parameters from Givens rotations,
% (produced using qrqrgivens), compute Q
% function Q = qrmakeqgiv(thetac,thetas)
% 
% thetac = cosine component of Givens rotation
% thetas = sin component of Givens rotation
%
% Q = orthogonal matrix formed by Givens rotations
```

```matlab
qrqtb.m
% 
% Given a matrix V containing Householder vectors as columns
% (produced using qrhouse), compute Q^H B, where Q is formed (implicitly)
% from the Householder vectors in V.
%
% function [B] = qrqtb(B,V)
% 
% B = matrix to be operated on
% V = matrix of Household vectors (as columns)
%
% output: B = Q^H B
```

```matlab
qrqtbgiv.m
% 
% Given thetac and thetas containing rotation parameters from Givens rotations,
% (produced using qrqrgivens), compute Q^H B, where Q is formed (implicitly)
% from the rotations in Theta.
%
% function [B] = qrqtbgiv(B,thetac,thetas)
%
% B = matrix to be opererated on
% thetac = cosine component of rotations from Givens rotations
% thetas = sine component of rotations from Givens rotations
%
% Output: B <-- Q^H B
```

```matlab
qrtheta.m
% 
% Given x and y, compute cos(theta) and sin(theta) for a Givens rotation
%
% function [c,s] = qrtheta(x,y)
%
% (x, y) = point to determine rotation
%
% c = cos(theta),   s=sin(theta)
```

```matlab
reducefree.m
% 
% Perform elimination on the free variables in a linear programming problem
%
% function [A,b,c,value,savefree,nfree] = reducefree(A,b,c,freevars)
% 
% A,b,c = parameters from linear programming problem
% freevars = list of free variables
%
% A,b,c = new parameters for linear programming problem (with free variables
%         eliminated)
% value = value of linear program
% savefree = tableau information for restoring free variables
% nfree = number of free variables found
```

```matlab
refltodir.m
% 
% Convert from a set of reflection coefficients kappa(1)...kappa(m)
% to FIR filter coefficients in direct form
%
% function a = refltodir(kappa)
%
% kappa = vector of reflection coefficients
%
% a = output filter coefficients = [1 a(1) a(2) ... a(m)]
```

```matlab
restorefree.m
% 
% Restore the free variables by back substitution
%
% function x = restorefree(inx,savefree,freevars)
% 
% inx = linear programming solution (without free variables)
% savefree = information from tableau for substitution
% freevars = list of free variables
% 
% x = linear programming solution (including free variables)
```

```matlab
rls.m
%
% Given a scalar input signal x and a desired scalar signal d,
% compute an RLS update of the weight vector h.
%
% This function must be initialized by rlsinit
%
% function [h,eap] = rls(x,d)
%
% x = input signal
% d = desired signal
%
% h = updated filter weight vector
% eap = (optional) a-priori estimation error
```

```matlab
rlsinit.m
%
% Initialize the RLS filter
%
% function rlsinit(m,delta)
```

```matlab
simplex1.m
% 
% Find the solution of a linear programming problem in standard form
%  minimize c'x
%  subject to Ax=b
%             x >= 0
%
% function [x,value,w] = simplex1(A,b,c,freevars)
% 
% A, b, c: system problem
% freevars = (optional) list of free variables in problem
%
% x = solution
% value = value of solution
% w = (optiona)l solution of the dual problem.
%    If w is used as a return value, then the dual problem is also solved.
%    (In this implementation, the dual problem cannot be solved when free
%    variables are employed.)
```

```matlab
sor.m
% 
% Produce an updated solution x to Ax = b successive over-relaxation
% A must be Hermitian positive definite
%
% function x = sor(A,x,b)
%
% A = input matrix
% x = initial solution
% b = right-hand side
% omega = relaxation parameter
% 
% Output x= updated solution
```

```matlab
sysidsvd.m
% 
% given a sequence of impulse responses in h
% identify a system (A,B,C)
%
% function [A,B,C] = sysidsvd(h,tol)
% h = impulse responses (a cell array)
% tol = (optional) tolerance used in rank determination
% ord = system order (overrides tolerance)
%
% (A,B,C) = estimated system matrix parameters
```

```matlab
testet.m
% testet.m
% Test the emission tomography code
% This script loads an image file, plots it, then calls the 
% code to test the tomographic reconstruction
```

```matlab
testirwls.m
% Test the irwls routine for a filter design problem
% After [Burrus 1994, p. 2934]
```

```matlab
testnn10.m
% Test the neural network on a pattern recognition problem
%
```

```matlab
tls.m
% 
% determine the total least-squares solution of Ax=b
%
% function x = tls(A,b)
%
% A = system matrix
% b = right-hand side
%
% x = TLS solution to Ax=b
```

```matlab
tohankel.m
% 
% Determine the matrix nearest to A which is Hankel and has rank r
% using the composite mapping algorithm
%
% function A = tohankel(A,r)
%
% A = input/output matrix
% r = desired rank
%
% Ouptut A = Hankel matrix with desired properties
% d = norm of difference between matrices
```

```matlab
tohanktoep.m
% 
% Determine the matrix nearest to A which is the stack
%  [ Hankel
%    Toeplitz ] 
% with rank r using the composite mapping algorithm
%
% function A = tohantoep(A,r)
%
% A = input matrix
% r = desired rank
%
% output: A = matrix with desired properties
% normlist = (optional) vector of errors
```

```matlab
tokarmarker.m
% 
% Given a linear programming problem in standard form
%  min       c'x
% subject to Ax = b,  x >= 0
%
% return new values of A and c in "Karmarker standard form"
%  min       c'x
% subject to Ax=0,  sum(x)=n, x >= 0
%
% function [Anew,cnew] = tokarmarker(A,b,c)
%
% (A,b,c) = matrices in standard form
%
% Anew, cnew = matrices in Karmarker standard form
```

```matlab
tostoch.m
% 
% Determine the matrix nearest to A which is stochastic using
% the composite mapping algorithm
%
% function A = tostoch(A)
% A = input matrix
%
% Output: A = nearest stochastic A
```

```matlab
tridiag.m
% 
% tridiagonalize the real symmetric matrix A
% 
% function [T,Q] = tridiag(A)
%
% A = matrix whose tridiagonal form is sought
%
% T = tridiagonal matrix
% Q = (optional) orthogonal transformation
```

```matlab
vitbestcost.m
% 
% Returns the best cost so far in the Viterbi algorithm
%
% function c = vitbestcost
```

```matlab
viterbi1.m
% 
% Run the Viterbi algorithm on the input for one branch
% Before calling this function, you must call initvit1
%
% function [p] = viterbi1(r)
%
% r = input value (scalar or vector)
%
% p = 0 if number of branches in window is not enough
% p = statenum on path if enough branches in window
%
% Call vitflush to clear out path when finished
```

```matlab
vitflush.m
% 
% Flush out the rest of the viterbi paths
%
% function [plist] = vitflush(termnode)
%
% termnode = 0 or list of allowable terminal nodes
%    If termnode==0, then choose path with lowest cost
%    Otherwise, choose path with best cost among termnode
%
% plist = list of paths from trellis
```

```matlab
warp.m
% 
% find the dynamic warping between A and B (which may not be of the
% same length)
%
% function [path] = warp(A,B)
%
% A = cells of the vectors, A{1}, A{2}, ..., A{M}
% B = cells of the vectors, B{1}, B{2}, ..., B{N}
%
% path = Kx2 array of (i,j) correspondence
```

```matlab
warshall.m
% 
% Find the transitive closure of the graph represented by the adjacency
% matrix A
%
% function Anew = warshall(A)
%
% A = adjacency matrix of graph
% 
% Anew = adjacency matrix for transitive closure of graph
```

```matlab
wavecoeff.m
% Coefficients for Daubechies wavelets
%
```

```matlab
wavetest.m
% Test the wavelet transform in matrix notation
```

```matlab
wavetesto.m
% Test the wavelet transform in wavelet notation (alternate indexing)
```

```matlab
wavetrans.m
%
% Compute the (nonperiodized) discrete wavelet traqnsform 
%
% function [C,ap] = wavetrans(c,coeff,J)
%
% c = data to be transformed
% coeff = wavelet transform coefficients
% J = number of levels of transform
%    If J is specified, then J levels of the transform are computed.  
%    Otherwise, the largest possible number of levels are used.
%
% C = transformed data
%     The output is stacked in C in wavelet-coefficient first order,
%     C = [d1 d2 ... dJ cJ]
% ap indexes the start of the coefficients for the jth level,
%     ap(j+1) indexes the start of the coefficients for the jth level,
%     except that ap(1) indicates the number of original datapoints
%
% This function simply stacks up the data for a call to the function wave
```

```matlab
wavetransper.m
%
% Compute the periodized discrete wavelet transform of the data
%
% function [C] = wavetransper(c,coeff,J)
%
% c = data to be transformed
% coeff = transform coefficients
% J indicates how many levels of the transform to compute.
%    If length(c) is not a power of 2, J must be specified.
%
% C = output vector
%    The output is stacked in C in wavelet-coefficient first order,
%    C = [d1 d2 ... dJ cJ]
```

```matlab
wftest.m
% Test the Wiener filter equalizer for a first-order signal and first-order
% channel
```

```matlab
zerorow.m
% 
% Zero a row by a series of Givens rotations
%
% function [B,U] = zerorow(B,f,U)
%
% B = matrix to have row zeroed
% f = vector of row indices that are zero on the diagonal
% U = (optional) rotation matrix
%
% B = modified matrix
% U = (optional) rotation matrix
```
