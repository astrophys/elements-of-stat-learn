<!--
Compile :
    pandoc -f markdown notes/somefile.md - -filter pandoc-crossref -t latex -o somefile.pdf

Notes:
    1. http://lierdakil.github.io/pandoc-crossref/
    #. On over/under braces : https://tex.stackexchange.com/a/132527/84495
-->


<!--
    YAML section
-->
---
title: Notes on Ch. 2 in Elements of Statistical Learning
author: Ali Snedden
date: 2021-12-07
abstract:
...
---
header-includes:
  - \hypersetup{colorlinks=true,
            urlcolor=blue,
            pdfborderstyle={/S/U/W 1}}
    \usepackage{mathtools}
    \usepackage{cancel}
---
\definecolor{codegray}{gray}{0.9}
\newcommand{\code}[1]{\colorbox{codegray}{\texttt{#1}}}
<!-- \let\overbracket\overbracket[0.2mm][0.6mm]{#1}      not sure that this works -->

\maketitle
\tableofcontents
\pagebreak


1. Read most of this chapter w/o taking notes in markdown, add notes later

Jargon
==========================
1. Recall Hastie uses $\hat{f}$ to denote predictions

2.1 Introduction
==========================

2.2 Variable Types and Terminology
==========================

2.3 Two Simple Approaches to Prediction, Least Squares and Nearest Neighbors
==========================
1. Linear Models and Least Squares
    a) Given vector of inputs, $X^{T} = (X_{1}, X_{2}, ..., X_{p})$, predicted output is:
       $$ \hat{Y} = \hat{\beta_{0}} + \sum^{p}_{j=1} X_{j} \hat{\beta}_{j}$$    {#eq:2.1}
    #) $\hat{\beta}$ is \emph{bias} in ML
    #) Transform $X^{T}$ to $X^{T} = (1, X_{1}, X_{2}, ..., X_{p})$, so we can package
       $\hat{\beta}_{0}$ into $\beta$ (could use Einstein notation here...)
       $$ \hat{Y} = X \hat{\beta}$$                                             {#eq:2.2}
    #) Usually, $\hat{Y}$ is a scalar, but generallizing : 
        #. $\hat{Y}$ can be a $K$-vector, $\beta$ would be $p \times K$ Matrix and 
    #) Least Squared fitting (most popular)
        $$RSS(\beta) = \sum^{N}_{i=1}(y_{i} - x_{i}^{T}\beta)^{2}$$             {#eq:2.3}
        Rewrite in matrix notation and do dimentional analysis
        $$RSS(\beta) = (\underbracket[0.2mm][0.6mm]{\bf y}_{\text{N}} - \overbracket[0.2mm][0.6mm]{\bf X}^{N\times p} \underbracket[0.2mm][0.6mm]{\beta}_{p})^{T}({\bf y} - {\bf X}\beta) $$  {#eq:2.4}
        Optimizing for $\beta$ (i.e. taking the derivative w/r/t $\beta$)
        $$ \frac{d}{d \beta} \big( RSS(\beta) = ({\bf y} - {\bf X} \beta)^{T}({\bf y} - {\bf X}\beta)\big) = 0$$ }
        That is hard, let's do a simple substitution, let $\beta' = ({\bf y} - {\bf X} \beta)$.
        Now above eqn becomes
        $$ \frac{d}{d \beta} \big( \beta'^{T} \beta' \big)= 2 \beta'^{T} \frac{d \beta'}{d \beta} = 0 $$ 
        Now substitute in for $\beta'$
        $$ \frac{d}{d \beta} \big( \beta'^{T} \beta' \big) = 2({\bf y} - {\bf X}\beta)^{T} \frac{d ({\bf y} - {\bf X}\beta)}{d \beta} = 2({\bf y} - {\bf X}\beta)^{T}(-{\bf X}) = 0$$ 
        Cancel -2 factor and take transpose across both sides 
        $$ {\bf X}^{T}({\bf y} - {\bf X}\beta) = 0$$                            {#eq:2.5}
        Now solve for $\beta$
        $$ {\bf X}^{T} {\bf y} - {\bf X}^{T} {\bf X}\beta = 0$$
        $$  + {\bf X}^{T} {\bf X}\beta = + {\bf X}^{T} {\bf y}$$
        $$   ({\bf X}^{T} {\bf X})^{-1} \big[{\bf X}^{T} {\bf X}\beta =  {\bf X}^{T} {\bf y} \big]$$
        $$   \beta =  ({\bf X}^{T} {\bf X})^{-1} {\bf X}^{T} {\bf y} \big]$$    {#eq:2.6}
    #) In Hasties' Figure 2.1, he uses categorical data where the points are orange or blue
        #. Blue : $Y = 0$ 
        #. Orange : $Y = 1$
    #) For the prediction, he cuts on $\hat{Y} = 0.5$, so 
        $$ \hat{G} = \left\{ \begin{array}{lr}
                                    \text{Orange} & \text{if  } \hat{Y} > 0.5 \\
                                    \text{Blue  } & \text{if  } \hat{Y}\leq 0.5
                                \end{array}
                       \right.
           $$     {#eq:2.7}
    #) The linear fit isn't perfect. Consider two possibilities
        #. Scenario 1 : The training data were generated by bivariate gaussian
                        distributions with uncorrelated components and different means
        #. Scenario 2 : The training data in each class came from a mixture of 10 
                        low-variance Gaussian distributions, with individual means dist. as
                        a gaussian.
        #. He has yet to tell us which it is.
#. Nearest-neighbor methods (2.3.2)
    a) The $k$-nearest neighbor fit (where $N_{k}(x)$ is neighborhood of $x$ with $k$
       closest points.
        $$ \hat{Y}(x) = \frac{1}{k} \sum_{x_{i} \in N_{k}(x)} y_{i}$$    {#eq:2.8}
    #) See \code{nearest\_neighbor.py} for implementation details
    #) When $k$ = 1, it corresponds to a \emph{Voronoi tessellation}
        #. NONE of the points are misclassified
        #. It is grossly overfitted
    #) Can't use minimizing errors (like least squares) to pick $k$. It would always pick   
       $k$ = 1
#. From Least Squares to Nearest Neighbors  (2.3.3)
    a) Least Squares fit = Low  Variance,   (potentially) high bias
        #. More suitable for Scenario 1
    #) Nearest Neighbors = High Varieance,  low bias
        #. More suitable for Scenario 2
    #) Data generated from
        #. BLUE : 10 means from bivariate Gassian distribution $N((1,0)^{T}, {\bf I})$
        #. ORANGE : 10 means from bivariate Gassian distribution $N((0,1)^{T}, {\bf I})$
        #. Then for BLUE (ORANGE) a mean was randomely selected from one of the 10 above 
           means ($m_{k}$), a 1/10 chance
        #. Then a point is selected from $N(m_{k}, {\bf I}/5)$
    #) Modifications and improvements
        #. Kernel methods use weights that decrease smoothly to zero from target point
           instead of the 0/1 effective weights used by $k$-nearest neighbors
        #. In high-dimensional spaces, the distance kernels are modified to emphasize
           some variable more than others
        #. Local regression fits linear models by locally weighted least squares rather 
           than fitting constants locally
        #. Linear models fit to a basis expansion of the original inputs allow 
           arbitrarily complex models
        #. Projection pursuit and neural network models consit of sums of non-linearly 
           transformed linear models.


2.4 Statistical Decision Theory
==========================
1. Background
    a) Conditional probability
        #. Given like $\text{Pr}(Y|X)$ 
        #. Said like "Probability of $Y$ given $X$"
        #. Example 1 : (from wiki), 
            * The probability of anyone having a cough is 
                $$\text{Pr}(cough) = 5%$$
              but, if a person is sick, that percentage may go up significantly, e.g.
                $$\text{Pr}(cough|sick) = 75%$$
        #. $\text{Pr}(Y|X)$ = Conditional Probility
        #. $\text{Pr}(Y)$   = Unconditional Probility
        #. If $\text{Pr}(Y|X) = \text{Pr}(Y)$ then $X$ and $Y$ are independent
        #. Example 2 : (from wiki), 
            * The probability of testing positive ($A$) if infected with dengue ($B$) is 90%, 
                $$\text{Pr}(A|B) = 90%$$
            * However there are high false positive rates in the dengue test, only 15% of
              positive tests are from people with dengue, ie
                $$\text{Pr}(B|A) = 15%$$
    #) Bayes' theorem
        $$\text{Pr}(A|B) = \frac{\text{Pr}(B|A) \text{Pr}(A)}{\text{B}}$$
        
            
#. Consider variables : 
    a) $X \in \mathbb{R}^{p}$ be a random input vector. 
    #) $Y \in \mathbb{R}$ be the random output variable.  
    #) $X$ and $Y$ are related by the joint probability distribution, $\text{Pr}(X,Y)$
    #) Want a function $f(X)$ to predict $\hat{Y}$ given $X$
    #) Introduce \emph{loss function} $L(Y,f(X)) = (Y - f(X))^{2}$ to penalize prediction
       error
    #) The Expected Prediction Error (EPE)
        $$ \text{EPE}(f) = \text{E}(Y - f(X))^{2}$$                             {#eq:2.9}
        $$ \text{EPE}(f) = \int [y - f(x)]^{2}\text{Pr}(dx,dy)$$                {#eq:2.10}
        #. QUESTION : Why is the $dx$, $dy$ w/in $\text{Pr}()$? I could understand better
                      if it was $\text{Pr}(X,Y)dx dy$?
        #. ANSWER   : It is awkward notation, maybe it should be
                      $\text{E}_{X} = \int x p(x) dx$? where $p(x)$ is the probability 
                      density function. He thinks Hastie means $\int (y-f(x))^{2} Pr(X,Y)dx dy$
    #) Conditionnig, i.e. factoring 
        $$\text{Pr}(X,Y) = \text{Pr}(Y|X)\text{Pr}(X)$$
       where
        $$\text{Pr}(Y|X) = \text{Pr}(Y,X)/\text{Pr}(X)$$
    #) Using this in eqn \ref{eq:2.10}
        $$ \text{EPE}(f) = \int [y - f(x)]^{2}\text{Pr}(dy|dx)\text{Pr}(dx)$$  
       some magic...I can understand or sort of see how you get expectation values, but still
       the math needs explained to me.
        $$ \text{EPE}(f) = \text{E}_{X} \text{E}_{Y|X} ([Y-f(X)]^{2}|X)$$       {#eq:2.11}
    #) Once again, we want to minimize eqn \ref{eq:2.11} to get $f(x)$
        $$ f(x) = \text{argmin}_{c} \text{E}_{Y|X} ([Y-c]^{2}|X=x)$$            {#eq:2.12}
       where $X=x$ is the conditional mean
    #) Solution is 
        $$ f(x) = \text{E}(Y|X=x)$$                                             {#eq:2.13}
    #) Can ask for the average of all the $y_{i}$'s with input $x_{i} = x$, but usually
       there is only one $y_{i}$ for each $x_{i}$
        $$ f(x) = \text{Average}(y_{i}|x_{i} \in N_{k}(x))$$                    {#eq:2.14}
       where $N_{k}(x)$ is the neighborhood containing $k$ points in $T$ closest to x.
       Some approximations : 
        #. Expectation is approx by averaging over sample data
        #. Conditioning (in sense of conditional probability) at a point is relaxed to 
           conditioning on some region "close" to the target point
    #) QUESTION : How do I get from eqns (\ref{eq:2.11} = \ref{eq:2.14})?
    #) ANSWER   : See the ANSWER above regarding eqn \ref{eq:2.10}
    #) Consider eqn \ref{eq:2.14} as $N$ (and $k$) go large. 
        #. $k/N \rightarrow 0$
        #. $\hat{f}(x) \rightarrow \text{E}(Y|X=x)$
        #. Issues
            * As dimensionality goes big, the rate of convergence goes to hell
            * We need to determine $\text{E}(Y|X)$ frome the data, 
    #) Consider linear regression w/in this framework
        #. Assuming that the regression function $f(x)$ is linear in its arguments
            $$ f(x) \approx x^{T}\beta $$                                       {#eq:2.15}
        #. Plugging in eqn \ref{eq:2.15} into eqn \ref{eq:2.9} 
            $$ \text{EPE}(f) = \text{E}(Y - x^{T}\beta)^{2}$$
            $$ \frac{d}{d\beta}\Big[\text{EPE}(f) = \text{E}(Y - x^{T}\beta)^{2}\Big]$$
            $$ 0 = 2x^{T}\text{E}(Y - x^{T}\beta)$$
            $$ . $$
            $$ . $$
            $$ . $$
            $$ \beta = [\text{E}(XX^{T}]^{-1}\text{E}(XY)$$                     {#eq:2.16}
            * Did not condition on X
            * Least squares solution in eqn \ref{eq:2.6} is basically replacing expectation
              values in eqn \ref{eq:2.16} with averages over the training data.
            * QUESTION : How?
        #. QUESTION : How do I go from above to eqn \ref{eq:2.16}?
        #. ANSWER   : I think the answer to the notation question for eqn \ref{eq:2.9} will 
                      help here
    #) Nearest neighbors and least squares approximate conditional expectations by averages.
        #. Differ in model assumptions 
            * Least squares assumes f(x) is well approximated by a globally linear
              functions
            * $k$-nearest neighbors assumes $f(x)$ is well approximated by a locally
              constant function
    #) Additive models assume
        $$ f(X) = \sum_{j=1}^{p}f_{j}(X_{j})$$                                  {#eq:2.17}
        #. Retains additivity of linear models, but $f_{j}$ is arbitrary
        #. Optimal estimate uses techniques like $k$-nearest neighbors to approximate
           \emph{univariate} conditional expectations \emph{simultaneously} for each of
           the coordinate functions.
        #. Sweeps away problems of estimating conditional expectation in high dimensions by 
           assuming additivity.
            * Often unrealistic
        #. QUESTION : I need some intuition here
        #. ANSWER   : He is breaking up the independence of each independent variable
                      s.t. there are no cross terms, e.g. leaving out $2xy$ from 
    #) Are we happy with eqn \ref{eq:2.11}?
        #. Consider replacing the quadratic $L_{2}$ loss function with a linear one
            $$ L_{1} = \text{E}|Y-f(X)| $$
        #. Solution becomes conditional median
            $$ \hat{f}(x) = \text{median}(Y|X=x)$$                              {#eq:2.18}
        #. Issues with continuity (i.e. derivative), will mention other loss functions 
           in the future
    #) How do we adapt loss functions for categorical data, $G$?
        #. Have to assume values for each category
        #. Loss function can be represented by $K\times K$ matrix ${\bf L}$ where 
            * $K = \text{card}(\mathcal{G})$. 
            * ${\bf L}$ is zero on diagonal nonnegative elsewhere - something special here
            * $L(k,l)$ is price paid to classify observation to class $\mathcal{G}_{k}$ as
              $\mathcal{G}_{l}$ 
            * QUESTION : What is special about a matrix with trace = 0?
            * ANSWER   : 'There is a special property for every kind of matrix'. It is
                         b/c it is a penalty matrix, which is why you aren't penalized for
                         correct answers. 
                + (Nathaneal suggests that it is a matrix full of ones with off diags being 0)
        #. QUESTION : Jargon here...
        #. ANSWER   : $card(\mathcal{G})$ is `cardinality', so 
        #. Assume \emph{zero-one} loss function where misclassifications are charged single
           unit, then 
            $$ \text{EPE} = \text{E}[L(G,\hat{G}(X))]$$                         {#eq:2.19}
           where expectation is taken w/r/t $\text{Pr}(G,X)$.  Use this to write
            $$ \text{EPE} = \text{E}_{X}\sum^{K}_{k=1} L[\mathcal{G}_{k}, \hat{G}(X)]\text{Pr}(\mathcal{G}_{k}|X)$$                         {#eq:2.20}
           minimize EPE pointwise
            $$ \hat{G}(x) = \text{argmin}_{g\in\mathcal{G}}\sum^{K}_{k=1} L(\mathcal{G}_{k}, g)\text{Pr}(\mathcal{G}_{k}|X)$$                         {#eq:2.21}
           with 0-1 loss function, simplifies to 
            $$ \hat{G}(x) = \text{argmin}_{g\in\mathcal{G}}[1-\text{Pr}(g|X=x)]$$   {#eq:2.22}
            $$ \hat{G}(x) = \mathcal{G}_{k} \text{ if } \text{Pr}(\mathcal{G}_{k}|X=x) = \text{max}_{g\in\mathcal{G}}\text{Pr}(g|X=x)$$   {#eq:2.23}
        #. eqn \ref{eq:2.23} is the Baye's classifier
            * QUESTION : To properly use, we need a-priori knowledge, if we don't have 
                         a-priori knowledge, what do we do?
            * ANSWER   : Yes, if you are handled a pile of data, you need to have some
                         knowledge before hand. Then he goes on to say here is how this
                         theory applies to different cases (e.g. linear regression)
           

2.5 Local Methods in High Dimensions
==========================
1. Curse of dimensionality
    a) Could find optimal conditional expectation by $k$-nearest neighbor averaging,
       but breaks down at high dimension
        #. QUESTION : How would we do this anyways?
        #. ANSWER   : In last section he was using a-priori knowledge, but if you don't 
                      have that, you can use $k$-nearest neighbors can approximate that
    #) Basically, as dimensionality increases, you have to ever increase the sampling of
       the space to adequately train your model
        #. E.g to sample 10% of the data (in 10D space) to form a local average, you need
           to cover $e_{p}(0.10) = r^{1/p} = 0.80$, or 80% of the range for EACH input
           variable
#. Consider a $N$ data points distributed uniformly over a $p$-dimensional unit ball
    #) Consider nearest-neighbor estimate at origin
    #) Median distance from origin to closest data point is 
        $$ d(p,N) = \Big(1 - \frac{1}{2}^{1/N} \Big)^{1/p}$$                    {#eq:2.24}
        #. For $N=500$, $p=10$, $d(p,N)\approx 0.52$, more than halfway to the 
           boundary of the sample
        #. This is a problem b/c the data points are closer to the edge than to another
           point
            * Estimating at the edge of a data set is fraught b/c it leads to extrapolation
              instead of interpolation.
            * This is intuitive, he has just formalized it.
#. He gives an example - in which I'm skipping some of the details.
    a) Consider 
        #. 1000 training examples $x_{i}$ generated uniformly on [-1,1]^{p}
        #. Assume true relationship between X and Y is :
            $$ Y = f(X) = e^{-8||X||} $$
        #. No measurement error.
            * QUESTION : What would it look like with measurement error?
            * ANSWER   : see eqn \ref{eq:2.26}
        #. Training set it $T$
        #. Compute \emph{mean squared error} (MSE), i.e.
            * The mean of the squares of the error
            * Squared to avoid adding in quadrature.
            * $E_{T}$ is the 'mean'
            * Summed over all the predictions ($\hat{y}_{i}$) of $y_{i}$ given $x_{i}$ 
            * $f(x)$ is the truth value. It is the underlying relation
        #. Problem is for estimating $f(0)$ in a deterministic case
    $$ \begin{aligned}
       \text{MSE}(x_{0}) & = \text{E}_{\mathcal{T}}[f(x_{0}) - \hat{y}_{0}]^{2} \\ 
                         & = \text{E}_{\mathcal{T}}[\hat{y}_{0} - \text{E}_{\mathcal{T}}(\hat{y}_{0})]^{2} + [\text{E}_{\mathcal{T}}(\hat{y}_{0}) - f(x_{0})]^{2} \\ 
                         & = \text{Var}_{\mathcal{T}}(\hat{y}_{0}) + \text{Bias}^{2}(\hat{y}_{0})
       \end{aligned}
    $$                                                                          {#eq:2.25}
    a) Let's do the derivation of eqn \ref{eq:2.25} following Wikipedia's derivation of
       the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error).
        #. Consider $n$ observations of variable $Y$ with n predictions $\hat{Y}$ 
           (think least-squares fit), the mean of the squared errors is :
            $$                                                                          
                \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{Y}_{i} - Y_{i})^{2}
            $$                                                                          
        #. This is really the expectation value of the the squared errors...
            $$                                                                          
            \begin{aligned}
                \text{MSE} & = \text{E}_{T}\Big[(f(x_{0}) - \hat{y}_{0})^{2}\Big] \\ 
                           & \text{  Adding 0}    \\
                           & = \text{E}_{\mathcal{T}}\Big[\overbrace{f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]}^{x} + \overbrace{\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y_{0}})^{2}}^{y}\Big] \\ 
                           & \text{  Expanding like any other quadratic}    \\
                           & = \text{E}_{\mathcal{T}}\Big[f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}] + \text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y_{0}})^{2}\Big] \\ 
                           & = \text{E}_{\mathcal{T}}\Big[\overbrace{\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big)^{2}}^{x^{2}} - \overbrace{2\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big)\big(\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y}_{0}\big)}^{2xy}+ \overbrace{\big(\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y_{0}}\big)^{2}}^{y^{2}}\Big] \\ 
                           & \text{  Recall that these are really integrals. Pull through $\text{E}_{\mathcal{T}}$}                   \\
                           & = \text{E}_{\mathcal{T}}\Big[\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big)^{2}\Big] - \text{E}_{\mathcal{T}}\Big[2\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big)\big(\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y}_{0}\big)\Big] + \text{E}_{\mathcal{T}}\Big[\big(\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y_{0}}\big)^{2}\Big] \\ 
                           & \text{Recall we have 1000 experiments at $x_{0}=0$, so the}\\
                           & \text{expectation value $\text{E}_{\mathcal{T}}(\text{E}_{\mathcal{T}}[\hat{y}_{0}])$ is a constant} \\
                           & \text{also expectation value $[f(x_{0}) - \text{E}_{\mathcal{T}}(\hat{y}_{0})]$ is a constant} \\
                           & = \text{E}_{\mathcal{T}}\Big[\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big)^{2}\Big] - 2\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big) \text{E}_{\mathcal{T}}\big(\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y}_{0}\big) +\text{E}_{\mathcal{T}}\Big[\big(\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y_{0}}\big)^{2}\Big] \\ 
                           & = \text{E}_{\mathcal{T}}\Big[\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big)^{2}\Big]
                             - 2\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big) \cancelto{0}{(\text{E}_{\mathcal{T}}[\text{E}_{\mathcal{T}}[\hat{y}_{0}]] - \text{E}_{\mathcal{T}}[\hat{y}_{0}])}
                             + \text{E}_{\mathcal{T}}\Big[\big(\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y_{0}}\big)^{2}\Big] \\ 
                           & = \text{E}_{\mathcal{T}}\Big[\big(f(x_{0}) - \text{E}_{\mathcal{T}}[\hat{y_{0}}]\big)^{2}\Big]
                             + \text{E}_{\mathcal{T}}\Big[\big(\text{E}_{\mathcal{T}}[\hat{y}_{0}] - \hat{y_{0}}\big)^{2}\Big] \\ 
                           & \text{The expected value of $\hat{y}_{0}$ should be the truth value}\\
                           & \text{so the second term is the \emph{bias} and the first term is the}\\ 
                           & \text{\emph{variance} (i.e. how much each predicted value differs form the truth)}\\
                           & = \text{Var}_{\mathcal{T}}(\hat{y}_{0}) + \text{Bias}^{2}(\hat{y}_{0})
            \end{aligned}
            $$
    #) eqn \ref{eq:2.25} is the \emph{bias-variance decomposition}
    #) QUESTION : What does $\text{E}_{\mathcal{T}}(\hat{y}_{0})$ mean?
    #) ANSWER   : Expectation value of ALL the predicted $\hat{y_{0}}$, N=1000
    #) QUESTION : For eqn \ref{eq:2.25} why is there an implicit sum for measuring MSE
                  for a single point. This is a point of confusion for me. They must be doing
                  1000 experiments at $x_{0} = 0$?
    #) ANSWER   : $x_{0}$ is at 0, but we have 1000 random samples over [-1,1]
    #) SUGGESTION : Follow 2nd derivation on Wikipedia. Less challenging to believe.
                  $\mathbb{E}$
    #) QUESTION : Is the logic in my last step reasonable?
    #) ANSWER   : He thinks so. Precision vs. Accuracy. I need to drop the expectation value
                  on my 2nd term on my last line.
#. Consider that the relation between $Y$ and $X$ is linear
    $$ Y = X^{T} \beta + \epsilon  $$                                           {#eq:2.26}
   where $\epsilon \sim N(0,\sigma^{2})$ and we fit with least squares.
    a) For arbitrary point $x_{0}$
            $$
            \begin{aligned}
                \hat{y}_{0} & = x^{T}_{0} \hat{\beta} \\
                                & = x^{T}_{0} \beta + \sum^{N}_{i=1}l_{i}(x_{0})\epsilon_{i}
            \end{aligned}
            $$                                           
       where $l_{i}(x_{0})$ is the $i$th element of ${\bf X (X^{T}X)^{-1}} x_{0}$. B/c 
       unbiased : 
            $$
            \begin{aligned}
                \hat{y}_{0} & = x^{T}_{0} \hat{\beta} \\
                            & = x^{T}_{0} \beta + \sum^{N}_{i=1}l_{i}(x_{0})\epsilon_{i}
            \end{aligned}
            $$                                                                   
       Deriving :
            $$
            \begin{aligned}
                \text{EPE}(x_{0}) & = \text{E}_{y_{0}|x_{0}} \text{E}_{\mathcal{T}}(y_{0} - \hat{y_{0}})^{2} \\
                                  & \text{I see how you sub in eqn \ref{eq:2.25}, but I'm}\\
                                  & \text{not sure what to do after that.}\\
                                  & \text{Says to make use of eqn 3.8}\\
                                  & = \text{Var}(y_{0}|x_{0}) + \text{E}_{\mathcal{T}}[\hat{y}_{0} - \text{E}_{\mathcal{T}}\hat{y}_{0}]^{2} + [E_{\mathcal{T}}\hat{y}_{0} - x_{0}^{T}\beta]^{2} \\
                                  & = \text{Var}(y_{0}|x_{0}) + \text{Var}_{\mathcal{T}}(\hat{y}_{0}) + \text{Bias}^{2}(\hat{y}_{0}) \\
                                  & = \sigma^{2} + \text{E}_{\mathcal{T}}x_{0}^{T}({\bf X}^{T}{\bf X}^{-1})x_{0}\sigma^{2} + 0^{2}\\
            \end{aligned}
            $$                                                                   {#eq:2.27}
        Incurred additional $\sigma^{2}$ in the prediction error b/c it is not deterministic
        like eqn \ref{eq:2.25}.  Bias is $0^{2}$ term and variance depends on $x_{0}$. If
        $N$ is large and $\mathcal{T}$ is selected at random. Assuming $\text{E}(X) = 0$
        and ${\bf X}^{T}{\bf X} \rightarrow N\text{Cov}(X)$ s.t.
            $$
            \begin{aligned}
                \text{E}_{x_{0}}\text{EPE}(x_{0}) & \sim \text{E}_{x_{0}} x_{0}^{T} \text{Cov}(X)^{-1}x_{0}\sigma^{2}/N + \sigma^{2} \\
                                  & = \text{Says to make use cyclic property of the trace}\\
                                  & = \text{trace}[\text{Cov}(X)^{-1}\text{Cov}(x_{0})]\sigma^{2}/N + \sigma^{2} \\
                                  & = \sigma^{2}(p/N) + \sigma^{2}
            \end{aligned}
            $$                                                                   {#eq:2.28}
        By using restrictions (e.g. $N$ is large, or $\sigma^{2}$ is small, variance growth
        is small (0 in deterministic case).

#. QUESTIONS : 
    a) QUESTION : Explain $E_{y_{0}|x_{0}}$ in eqn \ref{eq:2.27}. I thought $x_{0}$ is a 
                  single point at 0.
    #) ANSWER : THe expected value of over all values of $y_{0}$ while keeping $x_{0}$. It
                is the expectation value (recall the noise term, $\epsilon$) of all the $y_{0}$
                given the $x_{0}$
    #) QUESTION : The EPE and MSE look identical. Compare eqn \ref{eq:2.9} and eqn
                  \ref{eq:2.25} What's the difference? 
    #) ANSWER : Just that noise term.
    #) QUESTION : I see that in eqn \ref{eq:2.27}, EPE($x_{0}$) depends on $x_{0}$ while in
                  eqn \ref{eq:2.9} it depends on $f$, which is like $y$. WHat is going on here?
    #) ANSWER   : Probably just two different ways to look at the same thing?
    #) QUESTION : How does eqn 3.8 come into eqn \ref{eq:2.27} (exercise 2.5)?
    #) ANSWER   : Unsure see : [waxworksmath](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf) for the derivation he followed
    #) QUESTION : How is there no bias in eqn \ref{eq:2.28}?
    #) ANSWER   : B/c you are starting from eqn \ref{eq:2.27}, it is 0. He doesn't have a good
                  intuitive reason. Algebraicly he got it to disappear. 
    #) QUESTION : Why is there no bias in the linear case but there is in the deterministic case?

#. Discusses intuition from eqns \ref{eq:2.27} and \ref{eq:2.28}
    a) Whole variety of models that try to stradle strictly linear models and the highly
       flexible 1-nearest neighbor models.
        #. Models impose assumptions and biases to avoid exponential growth in complexity
           in high dimensions.

2.6 Statistical Models, Supervised Learning and Function Approximation
==========================
1. Goal to find useful approx of $\hat{f}(x)$ to the function $f(x)$
    a) Section 2.4, squared error loss leads to regression function $f(x) = \text{E}(Y|X=x)$.  
    #) Nearest neighbors can be thought of direct estimates of this conditional expectation.
       Fail in two ways
        #. If high dim on input space, nearest neighbors need not be close to target point
           and large errors result
        #. If special structure is known, this can reduce both bias and variance of
           estimates
#. 2.6.1 : A statistical model for the joint distribution $\text{Pr}(X,Y)$
    a) Suppose our data arose from the statistical model
        $$ Y = f(X) + \epsilon  $$                                           {#eq:2.29}
       where random error $\epsilon$ has $\text{E}(\epsilon) = 0$ and is independent of
       $X$. 
        #. Assume errors are independent and identically distributed
    #) QUESTION : I don't understand how models link to conditional probability
    #) ANSWER   : See : [Continuous random variables](https://en.wikipedia.org/wiki/Conditional_expectation) to get 
                  $$ f(x) = \text{E}[Y|X=x] = \int y \text{Pr}(Y|X)dy  $$
                  converting the variables on wikipedia appropriately
    #) For most input-output pairs $(X,Y)$, it will not have a deterministic relationship
       $Y = f(X)$
        #. Wow. I guess I assumed there would be underlying relationships (even if it
           can't be measured). Else, why do ML?
    #) Can have complex dependence of errors and variance on $X$
        #. Additive error model precludes this.
    #) Consider example of cardinality problem with two-class data
        #. Assume independent binary trials
        #. Outcome 0 has $p(X)$, Outcome 1 has $1-p(X)$
        #. If Y is 0-1 coded, then $\text{E}(Y|X = x) = p(x)$
#. 2.6.2 : Supervised Learning
    a) Assumes that the data fits a function, like eqn \ref{eq:2.29}
    #) Uses a \emph{teacher}
        #. Observe training set, $\mathcal{T} = (x_{i}, y_{i}), i = 1, ..., N$
        #. Learning algorithm modifies input / output relationship $\hat{f}$ in response to
           differences in $y_{i} - \hat{f}(x_{i})$
            * \emph{learning by example}
#. 2.6.3 : Function Approximation
    a) Mathematicians and statistitions view ML as function approximation in
       $\mathbb{R}^{p}$
        #. Data pairs ${x_{i}, y_{i}}$ are points in $(p+1)$-dimensional space
        #. $f(x_{i})$ in eqn \ref{eq:2.29} is in $p$-dimensional space
    #) Many approx. have associated with it $\theta$ that can be modified
        #. In $f(x) = x^{T}\beta$, $\theta = \beta$
        #. \emph{linear basis expansions}
            $$ f_{\theta}(x) = \sum_{k=1}^{K} h_{k}(x)\theta_{k}$$              {#eq:2.30}
            * $h_{k}$ : set of functions / transformation  of input vector $x$, e.g.
                + (e.g. polynomials, trig)
                + sigmoids transformation (used in neural networks)  
            $$ h_{k}(x) = \frac{1}{1+ e^{-x^{T}\beta_{k}}} $$                   {#eq:2.31}
                + Estimate params using RSS : 
            $$
                \text{RSS}(\theta) = \sum_{i=1}^{N} (y_{i} - f_{\theta}(x_{i}))^{2}
            $$                                                                  {#eq:2.32}
        #. Imagine the functions approx as a surface in $p+1$ space
            * See : Fig. 2.10
            * Verticle component is $y$. 
            * Try to get surface as close to observed points
                + Should point out overfitting is a problem here
                + Linear model and basis functions (assuming no hidden params) are
                  minimization problem. 
                + More complex problems require iterative methods or numerical optimization
                + Least squares is convenient, but isn't the only criteria and sometimes 
                  doesn't make sense.
    #) Maximum liklihood estimation
        #. Consider random sample, $y_{i}$, $i=1,...,N$ from density
           $\text{Pr}_{\theta}(y_{i})$
            * log-probability of observed sample is 
            $$
                L(\theta) = \sum_{i=1}^{N} \text{log} \text{Pr}_{\theta}(y_{i})
            $$                                                                  {#eq:2.33}
            * QUESTION : log base 10 or $e$? Probably doesn't matter.
        #. Assumes that most reasonable values of $\theta$ for which prob of observed
           sample is largest
            * Least squares in eqn \ref{eq:2.29} with $\epsilon \sim N(0, \sigma^{2})$ is 
              equivalent to maximum likelihood using 
            $$
                \text{Pr}(Y|X,\theta) = N(f_{\theta}(X), \sigma^{2})
            $$                                                                  {#eq:2.34}
            * QUESTION : I'm guessing $N$ in this case is the Gaussian function?
            * ANSWER   : Yes.
            * QUESTION : How do they go from above statement to eqn \ref{eq:2.34}
            * ANSWER   : Maybe it has to do with it is b/c in 
                         $Y = f_{\theta}(X) + \epsilon$ where
                         $\epsilong \sim N(0,\sigma^{2})$, you are just shifting your Gaussian
                         $f_{\theta}(X)$
            * Plug eqn \ref{eq:2.34} into eqn \ref{eq:2.33}
            $$
                \begin{aligned}
                L(\theta) & = \sum_{i=1}^{N} \text{log} \text{Pr}_{\theta}(y_{i}) \\
                          & = \sum_{i=1}^{N} \text{log} N(f_{\theta}(X), \sigma^{2}) \\
                          & = \sum_{i=1}^{N} \text{log}\Big(\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(y_{i} - f_{\theta}(X))^{2}}{2\sigma^{2}}}\Big) \\
                          & = \sum_{i=1}^{N}
                              \Big(
                                \text{log} \big( \frac{1}{\sigma \sqrt{2\pi}} \big) +
                                \text{log} \big( e^{-\frac{(y_{i} - f_{\theta}(X))^{2}}{2\sigma^{2}}}\big)
                              \Big) \\
                          & = \sum_{i=1}^{N}
                              \Big(
                                - \text{log} \big( \sigma \sqrt{2\pi} \big)
                                - \frac{(y_{i} - f_{\theta}(X))^{2}}{2\sigma^{2}}\big)
                              \Big) \\
                          & = \sum_{i=1}^{N}
                              \Big(
                                - \text{log} \big( \sigma \big)
                                - \text{log} \big((2\pi)^{\frac{1}{2}}) \big)
                                - \frac{(y_{i} - f_{\theta}(X))^{2}}{2\sigma^{2}}\big)
                              \Big) \\
                          & \text{Pull out first two (constant) terms, incurring a factor of N} \\
                          & =
                                - N \text{log} \big( \sigma \big)
                                - \frac{N}{2} \text{log} \big(2\pi \big)
                                + \sum_{i=1}^{N} - \frac{(y_{i} - f_{\theta}(X))^{2}}{2\sigma^{2}} \\
                          & =
                                - N \text{log} \big( \sigma \big)
                                - \frac{N}{2} \text{log} \big(2\pi \big)
                                - \frac{1}{2\sigma^{2}} \sum_{i=1}^{N} (y_{i} - f_{\theta}(X))^{2}
                \end{aligned}
            $$                                                                  {#eq:2.35}
            Last term in eqn \ref{eq:2.35} is the $\text{RSS}(\theta)$ up to a negative
            multiplier.
            * QUESTION : Implied dimension? Gaussian normalization depends on dimensionality
            * ANSWER   : He agrees
    #) Consider a multinomial likelihood of regression function $\text{Pr}(P|X)$ for a
       qualitative output $G$. 
        #. Conditional probability of each class given $X$ : 
            $$ \text{Pr}(G = \mathcal{G}_{k}|X = x) = p_{k,\theta}(x), k = 1, ..., K $$
           yields log-likelihood
            $$ L(\theta) = \sum_{i=1}^{N} \text{log}(p_{g_{i},\theta}(x_{i}) $$ {#eq:2.36}
        #. QUESTION : Is K the number of samples? or classes? - I think classes
        #. ANSWER   : Classes. $G$ is the set of classes, $K$
        
2.7 Structured Regression Models
==========================
1. Observation
    a) Local methods (e.g. nearest-neighbors) focus on estimating function at a point
        #. Screwed in high dimensions (see curse of dimensionality)
        #. Inappropriate when there is a structured approach available (i.e. data follows a 
           functions)
#. Difficulty of the Problem
STOPPED HERE - Moved on to Ch3 for sake of expedience...


2.8 Classes of Restricted Estimators
==========================
  
2.9 Model Selection and the Bias-Variance Tradeoff
==========================
<!--
                           & = \text{E}_{T}\Big[(\hat{Y} - E[\hat{Y}] + E[\hat{Y}] - Y)(\hat{Y} - E[\hat{Y}] + E[\hat{Y}] - Y)\Big] \\ 
                           & = \text{E}_{T}\Big[\hat{Y}^{2} - \hat{Y}E[\hat{Y}] + \hat{Y}E[\hat{Y}] - \hat{Y}Y -E[\hat{Y}]\hat{Y} + (E[\hat{Y}])^{2} - (E[\hat{Y}])^{2} + E[\hat{Y}]Y + \\
                           &   \hspace{8mm} E[\hat{Y}]\hat{Y} - (E[\hat{Y}])^{2} + (E[\hat{Y}])^{2}  - E[\hat{Y}]Y - Y\hat{Y} + Y E[\hat{Y}] - Y E[\hat{Y}] + Y^{2}\Big]        \\
            \end{aligned}
            $$                                                                          
           Rearrange Terms
            $$                                                                          
            \begin{aligned}
                \text{MSE} & = \text{E}\Bigg[\Big(\hat{Y}^{2} - \hat{Y}E[\hat{Y}] - \hat{Y}E[\hat{Y}] + (E[\hat{Y}])^{2}\Big)  \Bigg]
            \end{aligned}
            $$                                                                          
            COMPLETE LATER - Need to determine exactly what each term EXACTLY means
-->
    


    


#. Questions
    a) $E_{x}$ = expectation value?
        #. I think so
    #) I don't understand the notation in eqn \ref{eq:2.10}
    #) I don't understand the math going from eqn \ref{eq:2.10} to eqn \ref{eq:2.11}?
       How do I handle dealing with differentials in joint probability distribution
       functions?
    #) How do we go from eqn \ref{eq:2.12} to eqn \ref{eq:2.13}
    
#. To Do
    a) Work on derivation fo eqn \ref{eq:2.9} to eqn \ref{eq:2.14}
    a) Work on derivation fo eqn \ref{eq:2.16}


2.5 Local Methods in High Dimensions
==========================

2.6 Statistical Models, Supervised Learning and Function Approximation
==========================
1. Review
    a) Goal
        #. Find approx of $\hat{f}(x)$ to underlying function $f(x)$
    #) Previously
        #. 
    
#. A Statistical Model for the Joint Distribution Pr(X, Y)



Exercises
==========================
1. Exercises that I think would be worth doing:
    a) 2.2 - Done. See \code{bayes.py}
    #) 2.3
    #) 2.5
    #) Either 2.7 OR 2.9, not both
    
