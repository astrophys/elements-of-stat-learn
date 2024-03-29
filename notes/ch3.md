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
title: Notes on Ch. 3 in Elements of Statistical Learning
author: Ali Snedden
date: 2022-02-24
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

3.1 Introduction
==========================
1. Easy to understand linear models, then we can generalize to non-linear models

3.2 Variable Types and Terminology
==========================
1. From Ch2, refresher on linear models :
    a) Given input vector $X^{T} = (X_{1}, X_{2}, \ldots, X_{p})$, with predicted output
       $Y$, the model is :
        $$
            f(X) = \beta_{0} + \sum_{j=1}^{p} X_{j} \beta_{j}
        $$                                                                      {#eq:3.1}
       where $\beta$ is unknown and $X_{j}$ can be :
        #. quantitative inputs
        #. transformations of quantitative inputs (e.g. log, $\sqrt{}$)
        #. basis expansions, e.g. $X_{2} = X_{1}^{2}$, $X_{3} = X_{1}^{3}$
        #. numeric / dummy coding for cardinal data.
        #. interactions between variables, e.g. $X_{3} = X_{1} \dot X_{2}$
    #) Training data is of format $(x_{1}, y_{1}) \ldots (x_{N}, y_{N})$ where vector feature
       of measurements is $x_{i} = (x_{i1}, x_{i2}, \ldots, x_{ip})$.   
        #. Note $p$ is the number of variables, $N$ is the number of measurements of $p$
           variables
    #) Most common estimate is by method of \emph{least squares} in which we pick the 
       coefficients $\beta = (\beta_{0}, \beta_{1}, \ldots, \beta_{p})$
        $$
          \begin{aligned}
            \text{RSS} & = \sum_{i=1}^{N} (y_{i} -f(x_{i}))^{2} \\
                & = \sum_{i=1}^{N} \Big(y_{i} - \beta_{0} - \sum_{j=1}^{p} x_{ij}\beta_{j}\Big)^{2}
          \end{aligned}
        $$                                                                      {#eq:3.2}
        #. Valid if $(x_{i}, y_{i})$ are 
            * independent random draws OR
            * even if $x_{i}$'s aren't randomly draw, it holds if $y_{i}$'s are 
              conditionally independent given inputs $x_{i}$
            * QUESTION : What does this EXACTY mean? 
            * ANSWER   : $y_{1}$ doesn't depend on $y_{2}$, but they CAN be dependent on 
                         $x_{1}$ even if $x_{i}$ follows some function. This makes sense
        #. Fig. 3.1 - Visualize as a hyperplane in $p + 1$ dimensional space where the $+1$
           is the $y$ dependent variable.
            * Makes no comment about validity of fit
    #) Re-hashing derivation of RSS from Ch2:
        $$
          \begin{aligned}
            \text{RSS} & = ({\bf y} - {\bf X}\beta)^{T} ({\bf y} - {\bf X}\beta)
          \end{aligned}
        $$                                                                      {#eq:3.3}
       Eqn \ref{eq:3.3} is a quadratic with $p+1$ matrix (recall the $+1$ is for the intercept).
       Minimize eqn \ref{eq:3.2} as a function of $\beta$. I did a full derivation in Ch 2.
        $$
          \begin{aligned}
            \frac{\partial \text{RSS}}{\partial \beta} & = -2 {\bf X}^{T}({\bf y} - {\bf X}\beta) \\
            \frac{\partial^{2} \text{RSS}}{\partial \beta \partial \beta^{T}} & = -2 {\bf X}^{T}({\bf X} \\
          \end{aligned}
        $$                                                                      {#eq:3.4}
       Assume ${\bf X}$ is full rank (i.e. each is independent). Setting first derivative $=0$,
        $$
          \begin{aligned}
            -2 {\bf X}^{T}({\bf y} - {\bf X}\beta) = 0 \\
            {\bf X}^{T}({\bf y} - {\bf X}\beta) = 0 \\
          \end{aligned}
        $$                                                                      {#eq:3.5}
       Rearranging eqn \ref{eq:3.5}
        $$
          \begin{aligned}
            \hat{\beta} = ({\bf X}^{T} {\bf X})^{-1}{\bf X}^{T}{\bf y}
          \end{aligned}
        $$                                                                      {#eq:3.6}
       The predicted values, based off the model $\beta$ learned from the training set.
        $$
          \begin{aligned}
            \hat{y} = {\bf X}\hat{\beta} = {\bf X}({\bf X}^{T} {\bf X})^{-1}{\bf X}^{T}{\bf y}
          \end{aligned}
        $$                                                                      {#eq:3.7}
       From eqn \ref{eq:3.7} we call ${\bf H} = {\bf X}({\bf X}^{T} {\bf X})^{-1}{\bf X}^{T}$
       the 'hat' matrix b/c it puts a 'hat' on ${\bf y}$. Aka the `projection matrix'
        #. QUESTION : Shouldn't eqn \ref{eq:3.7} have ${\bf X}^{T}$. See text below
           eqn 2.6?
        #. ANSWER : He doesn't know why. Maybe switched from column / row vector. Maybe
                    it is b/c in eqn \ref{eq:3.7} it is a matrix instead of a vector.
                    Suboptimal notation.
    #) Figure 3.2     
        #. Geometrical representation of least squares estimate
        #. Denote column vectors ${\bf X}$ as ${\bf x}_{0}, {\bf x}_{1}, \ldots, {\bf x}_{p}$
           with ${\bf x}_{0} = 1$
    #) Minimize $\text{RSS}(\beta)$ by choosing $\hat{\beta}$ so that residual vector 
       ${\bf y} - {\bf \hat{y}}$ is \emph{orthogonal} to this subspace
        #. QUESTION : Explain this, I don't have intuition on why it is 'orthoganol'
        #. ANSWER   : B/c $\hat{y}$ is in the basis set of $x_{i}$. When you multiply
                      a matrix by a vector, the resultant vector has to be in the space
                      from $x_{i}$ with $\beta_{i}$ being the coefficients of the basis set
                      Truth ($y$) is NOT going to exactly be in the column space.
                      The reason it is \emph{orthogonal} is b/c projection is the CLOSEST point 
                      to the space (recall we took the minimum).
    #) If not all columns in ${\bf X}$ are linearly independent, it is not full rank
        #. E.g. ${\bf x_{2}} = 3 {\bf x_{3}}$
        #. Then ${\bf X}^{T}{\bf X}$ is singular
        #. In this case fitted values ${\bf \hat{y}}$ are still a projection of ${\bf y}$ 
           onto space ${\bf X}$ (see Figure 3.2)
        #. Most software try to detect this condition and automatically try removing them.
    #) Rank deficiencies also occur in signal / image processing where the number of inputs $p$
       can exceed the number of training cases $N$
        #. Use regulation or filtering
#. Let's make some assumptions about the training data (previously we did not)
    a) Consider (assume?)
        #. $y_{i}$ are uncorrelated and has constant variance $\sigma^{2}$
            * QUESTION : Confusing, b/c they would be correlated if $x_{i}$ follow some 
                         underlying function. Maybe he means the act of measuring $y_{1}$
                         has no effect on $y_{2}$
            * ANSWER : Already answered near eqn \ref{eq:3.2}
        #. $x_{i}$ are fixed (i.e. non-random)
    #) Variance-covariance mastrix of least squres params $(\beta)$ is derived from
       eqn \ref{eq:3.6}.
       Starting with eqn \ref{eq:3.6}.
        $$
          \begin{aligned}
            \hat{\beta} = ({\bf X}^{T} {\bf X})^{-1}{\bf X}^{T}{\bf y}
          \end{aligned}
        $$
       We know that generally the equation for variance (single variate) is
        $$
          \begin{aligned}
            \text{Var}(X) & = \text{E}[(X - \mu)]^{2} \\
                & = \frac{1}{N - 1} \sum_{i=0}^{N} [(X_{i} - \mu)]^{2} \\
          \end{aligned}
        $$
       However we don't have a univarite ${\bf X}$. Consider that we are trying to
       find the variance in the predicted $\beta$. Also recall that above we said ${\bf X}$
       is fixed, so only the ${\bf y}$ component of eqn \ref{eq:3.6} can vary. 
        $$
          \begin{aligned}
            \text{Var}(\hat{\beta}) & = \text{E}[(\hat{\beta} - \overline{\hat{\beta}}^{2})] \\
            & \vdots \\
            & \text{and by some not-so-easy magic}    \\
            & \vdots \\
            \text{Var}(\hat{\beta}) & = ({\bf X}^{T}{\bf X})
          \end{aligned}
        $$
        #. QUESTION : Is my derivation correct? What is exactly $\overline{\hat{\beta}}$. Is it 
                      $\text{mean}(\hat{\beta})$ or is it $\text{mean}(\hat{\beta_{i}})$ ?
        #. ANSWER   : $\text{E}(\hat{\beta})$ is a vector. Look at his derivation
        $$
          \begin{aligned}
            \text{Var}(\hat{\beta}) = ({\bf X}^{T}{\bf X})^{-1} \sigma^{2}
          \end{aligned}
        $$                                                                      {#eq:3.8}
    #) Estimate variance (think standard deviation here) $\sigma^{2}$
        $$
          \begin{aligned}
            \hat{\sigma}^{2} = \frac{1}{N - p - 1} \sum^{N}_{i=1} (y_{i} - \hat{y_{i}})^{2}
          \end{aligned}
        $$
       where $N - p - 1$ is used instead of $N$ to make $\hat{\sigma}^{2}$ an unbiased 
       estimate of $\sigma^{2} : \text{E}(\hat{\sigma}^{2}) = \sigma^{2}$
        #. Note that this is different from estimating the std. dev. of variance of a
           single mean b/c the predicted value of y isn't a constant
#. Draw inferences about model
    a) additional assumptions
        #. eqn \ref{eq:3.1} is correct model of mean
        #. conditional expectation of $Y$ is linear in $X_{1}, \ldots, X_{p}$
        #. deviations of $Y$ about expectation are additive and Gaussian
        $$
          \begin{aligned}
            Y & = \text{E}(Y|X_{1}, \ldots, X_{p}) + \epsilon \\
              & = \beta_{0} + \sum_{j=1}^{p} X_{j} \beta_{j} + \epsilon
          \end{aligned}
        $$                                                                      {#eq:3.9}
           where $\epsilon \sim N(0, \sigma^{2})$ (adds variance, doesn't shift mean
           on average)
            * QUESTION : Shouldn't eqn \ref{eq:3.9} have $\epsilon_{j}$?
            * ANSWER   : He thinks he wasn't careful with is paranthesis
    #) Starting with eqn \ref{eq:3.9}
        $$
          \begin{aligned}
              Y & = \beta_{0} + \sum_{j=1}^{p} X_{j} \beta_{j} + \epsilon \\
                & = {\bf X} \beta + N(0, \sigma^{2}) \\
                & \quad \text{ This is equivalent to `shifting' the Gaussian by ${\bf X}\beta$}\\
                & = N({\bf X} \beta, \sigma^{2})      \\
                & \quad \text{ Plugging above result into eqn \ref{eq:3.6}}      \\
              \hat{\beta} & = ({\bf X}^{T}{\bf X})^{-1}{\bf X}^{T} {\bf y}       \\
                & = ({\bf X}^{T}{\bf X})^{-1}{\bf X}^{T} {\bf y}                 \\
                & = ({\bf X}^{T}{\bf X})^{-1}{\bf X}^{T} N({\bf X} \beta, \sigma^{2})   \\
                & = N(\cancelto{\bf I}{({\bf X}^{T}{\bf X})^{-1}({\bf X}^{T} {\bf X})} \beta, ({\bf X}^{T}{\bf X})^{-1}{\bf X}^{T}\sigma^{2}) \\
                & = N(\beta, ({\bf X}^{T}{\bf X})^{-1}{\bf X}^{T}\sigma^{2})            \\
                & \quad \text{ Missing factor of $({\bf X}^{T})^{-1}$ for $\sigma$ term}\\
                & \quad \text{ I suspect the error comes from 2nd line}                 \\
          \end{aligned}
        $$                                                                      {#eq:3.10}
        #. QUESTION : Resolve missing term above?
        #. ANSWER   : Go back to his derivation previously of 
    #) Chi-squared distribution 
        $$
          \begin{aligned}
            (N - p - 1)\hat{\sigma}^{2} \sim \sigma^{2} \chi^{2}_{N-p-1}
          \end{aligned}
        $$                                                                      {#eq:3.11}
       Use eqn \ref{eq:3.10} and \ref{eq:3.11} to get distributional properties for params
       $\beta_{j}$
        #. E.g. hypothesis testing and confidence intervals
#. \emph{Z-score} for $\beta_{j} = 0$
        $$
          \begin{aligned}
            z_{j} = \frac{\hat{\beta_{j}}}{\hat{\sigma}\sqrt{v_{j}}}
          \end{aligned}
        $$                                                                      {#eq:3.12}
   where $v_{j}$ is the $j$th diagonal element of $({\bf X}^{T}{\bf X})^{-1}$
    a) Null hypothesis 
        #. $\beta_{j} = 0$
        #. $z_{j}$ is distributed as $t_{N-p-1}$ ($t$ distribution w/ $N-p-1$ deg of freedom)
        #. If $\hat{\sigma}$ with known value $\sigma$, $z_{j}$ becomes standard gaussian dist
            * Tail properties of $t$-dist is negligible at large sample sizes
#. F statistic
    a) Consider categorical variable with $k$ levels
        #. Want to test if coeffs of dummy vars used to represent levels can be set to 0
        #. Enter the $F$ statistic
        $$
          \begin{aligned}
            F & = \frac{(\text{RSS}_{0} - \text{RSS}_{1})/(p_{1} - p_{0})}{\text{RSS}_{1}/(N-p_{1}-1}) \\
          \end{aligned}
        $$                                                                      {#eq:3.13}
           where 
            * $\text{RSS}_{1}$ : RSS for the least squares fit of the bigger model
                                 with $p_{1} + 1$ params
            * $\text{RSS}_{0}$ : RSS for the least squares fit of the smaller model with
                                 $p_{0} + 1$ params, with $p_{1} - p_{0}$ params
                                 constrained to 0
        #. Measures change in RSS per additional param in bigger model
    #) Exercise 3.1 shows that $z_{j}$ in eqn \ref{eq:3.12} are equivalent to the $F$ 
       statistic for dropping a single coeff $\beta_{j}$ from model.
    #) Can isolate $\beta_{j}$ in eqn \ref{eq:3.10} to obtain a $\text{1 - 2 $\alpha$}$
       confidence interval
        $$
          \begin{aligned}
            (\hat{\beta}_{j} - z^{(1-\alpha)}v_{j}^{\frac{1}{2}} \hat{\sigma}, \hat{\beta}_{j} + z^{(1-\alpha)}v_{j}^{\frac{1}{2}} \hat{\sigma})
          \end{aligned}
        $$                                                                      {#eq:3.14}
       where $z^{1-\alpha}$ is the $1-\alpha$ percentile of the normal distribution
        $$
          \begin{aligned}
            z^{(1-0.025)} & = 1.96    \\
            z^{(1-0.05)}  & = 1.645   \\
          \end{aligned}
        $$
        #. Typically reported as $\hat{\beta} \pm 2 \cdot \text{se}(\hat{\beta})$
            * Approx. a 95% confidence interval
            * QUESTION : What is 'se'?
            * ANSWER   : 'standard error'
    #) CI for entire vector $\beta$
        $$
          \begin{aligned}
            C_{\beta} = \{\beta|(\hat{\beta}-\beta)^{T}{\bf X}^{T}{\bf X}(\hat{\beta} - \beta) \leq \hat{\sigma}^{2} \chi_{p+1}^{2(1-\alpha)}\}
          \end{aligned}
        $$                                                                      {#eq:3.15}
        #. QUESTION : Typo for $\chi$ in eqn \ref{eq:3.15}? Explain?
        #. ANSWER   : Nope $l = p+1$
#. 3.2.1 - Example : Prostate Cancer
    a) Data - Stamey et al. 1989
        #. Variables 
            * log cancer volume (lcavol)
            * log prostate weight (lweight)
            * age
            * log of benign protatic hyperplasia (lbph)
            * seminal vesicle invasion (svi)
            * log of capsular penetration (lcp)
            * Gleason score (gleason) 
            * percent of Gleason scores 4 or 5 (pgg45)
        #. Fig. 1.1 on p3 illustrates the data
    #) Let's implement this in Python!
        #. Procedure 
            * Standardize data 
                + Center on mean
                + Make unit variance
            * Randomly split into Training set (67) and Test set (30)
            * Apply least squares to estimate the coefficients
            * Compute Z-scores \ref{eq:3.12}
        #. See src/prostate_cancer.py
            * I didn't get the correct value for the Intercept...
    #) Z-score
        #. Measure's the effect of dropping variable from model
        #. Absolute value $>2$ is approximately significant at 5% level.
        #. lcavol, lweight and svi are significant (Table 3.2)
        #. Can test for exclusion of a number of terms at once using
           $F$-statistic (\ref{eq:3.13})
    #) Dropping age, lcp, gleason and pgg45
        $$
          \begin{aligned}
            F & = \frac{(\text{RSS}_{0} - \text{RSS}_{1})/(p_{1} - p_{0})}{\text{RSS}_{1}/(N-p_{1}-1}) \\
          \end{aligned}
        $$                                                             {#eq:3.16}
#. 3.2.2 - Gauss-Markov Theorem
    a) Famouse results : least squares estimates $\beta$ have smallest varience among all 
                         \emph{linear} unbiased estimates.
        #. Leads us to ridge regression later
        #. Consider any linear combination of params $\theta = a^{T}\beta$
            * $f(x_{0}) = x_{0}^{T}\beta$, a linear function is of this form
        #. Now the the \emph{prediction} of $\theta$ is $\hat{\theta}$
        $$
          \begin{aligned}
            \hat{\theta} = a^{T} \hat{\beta} = a^{T} ({\bf X}^{T} {\bf X})^{-1} {\bf X}^{T} {\bf y}
          \end{aligned}
        $$                                                             {#eq:3.17}
            * If ${\bf X}$ is fixed, $\hat{\theta}$ is a linear function
              (${\bf c}_{0}^{T}{\bf y}$), then $a^{T} \hat{\beta}$ is unbiased.
        $$
          \begin{aligned}
            \text{E}(a^{T} \hat{\beta})  = \\
                & \text{Subsitute $\hat{\beta} = ({\bf X}^{T} {\bf X})^{-1}{\bf X}^{T}{\bf y}$} \\
                & = \text{E}(a^{T}({\bf X}^{T} {\bf X})^{-1} {\bf X}^{T} {\bf y}) \\
                &   \text{$a^{T}$ and ${\bf X}$ is a constant, so pull it all out of expectation value}  \\
                & = a^{T}({\bf X}^{T} {\bf X})^{-1} {\bf X}^{T} \text{E}({\bf y})   \\
                &   \text{Expected value of ${\bf y}$ is ${\bf X}\beta$}   \\
                & = a^{T}\cancelto{\mathbb{I}}{({\bf X}^{T} {\bf X})^{-1} {\bf X}^{T} {\bf X}} {\beta})    \\
                &   \text{I'm confused here about exactly how and where ${\bf X} {\beta}$ come from }\\
                & = a^{T}\beta
            \text{E}(\theta) & = \theta
                &   \text{Thus E$(\theta) = \theta$}\\
          \end{aligned}
        $$                                                             {#eq:3.18}
         Eqn \ref{eq:3.18} proves that it is unbiased. The key assumption is that the linear
         model is correct (this is KEY). Thus the expectation of the prediction should equal
         the actual values.
        #. Gauss-Markov theorem states : if we have any other linear estimator
           $\tilde{\theta} = {\bf c}^{T} {\bf y}$ that is unbiased for $a^{T} \hat{\beta}$,
           i.e. $\text{E} ({\bf c} {\bf y}) = a^{T} \beta$ then
        $$
          \begin{aligned}
            \text{Var}(a^{T}\hat{\beta}) \le \text{Var}({\bf c}^{T}{\bf y})
          \end{aligned}
        $$                                                             {#eq:3.19}
        #. Consider mean squared error ($\text{MSE}(\hat{\theta})$) for any estimator, where
           $\theta$ is the truth.
        $$
          \begin{aligned}
            \text{MSE}(\tilde{\theta}) & = \text{E}(\tilde{\theta} - \theta)^{2}            \\ 
                & = \text{Var}(\tilde{\theta}) + [\text{E}(\tilde{\theta})  - \theta]^{2}
          \end{aligned}
        $$                                                             {#eq:3.20}
           Extending eqn \ref{eq:3.20} to the linear model.
        $$
          \begin{aligned}
            \text{MSE}(\tilde{\theta}) & = \text{Var}(\tilde{\theta}) + [\text{E}(\tilde{\theta})  - \theta]^{2} \\
                & = \text{Var}(\tilde{\theta}) + [\theta - \theta]^{2}    \\
                & = \text{Var}(\tilde{\theta}) \\
          \end{aligned}
        $$
           The upshot is that the Gauss-Markov theorem implies that the least
           squares estimator has the smallest mean squared error of ALL linear estimators
           with NO BIAS.
        #. However, there may still be a BIASED estimator with lower MSE...
            * Trades off some bias for less variance
            * Best models trade this off. Enter ridge regression and such
        #. MSE is intimately related to prediction accuracy.  
        $$
          \begin{aligned}
            Y_{0} = f(x_{0}) + \epsilon_{0}
          \end{aligned}
        $$                                                             {#eq:3.21}
           The expected prediction error (see eqn \ref{eq:3.20}) for an estimate
           $\tilde{f}(x_{0}) = x_{0}^{T} \hat{\beta}$
        $$
          \begin{aligned}
            \text{E}(Y_{0} - \tilde{f}(x_{0}))^{2} & = \sigma^{2} + \text{E}(x_{0}^{T}\tilde{\beta} - f(x_{0}))^{2} \\
                & = \sigma^{2} + \text{MSE}(\tilde{f}(x_{0}))
          \end{aligned}
        $$                                                             {#eq:3.22}
          where $\sigma^{2}$ is the variance of the new observation $y_{0}$

        #. Q/A : 
            * Q : Where does the additional ${\bf X}$ come in from the expectation value?
            * A : ${\bf X}$ is constant and $\text{E}({\bf y}) = {\bf X}\beta$
            * Q : Is $a^{T}$ just ${\bf X}$?
            * A : Yes
            * Q : parameters in $\theta$ don't have to be a linear function, correct?
            * A : Yes
            * Q : Is \ref{eq:3.18} unbiased b/c the expectation value did NOT pick up an 
                          extra term? Is it unbiased b/c (${\bf c}_{0}^{T}{\bf y}$) is linear?
            * A : Yes
            * Q : Not sure why the variances aren't just equal?
            * A : The KEY point is that the estimator $\hat{\theta}$ is the Least 
                          squares estimate, where $\tilde{\theta}$ can be ANY other estimator.
            * Q : In eqn \ref{eq:3.22} what is $y_{0}$?  $x_{0}^{T} \hat{\beta}$?


#. 3.2.3 - Multiple Regression from Simple Univariate Regression
    a) Multiple Linear Regression Model
        #. Recall from eqn \ref{eq:3.1} that ${\bf X} = \{X_{1}, X_{2}, \ldots, X_{p}\}$
        #. Called multiple linear regression model when $p > 1$
        #. Best understood for \emph{univariate} model, i.e. $p = 1$
    #) Consider univariate model \emph{without} intercept
        $$
          \begin{aligned}
            Y = X \beta + \epsilon
          \end{aligned}
        $$                                                             {#eq:3.23}
       Rearranging eqn \ref{eq:3.6} for a single $X$, the and $N$ measurements, the 
       least squares estimate and residuals
        $$
          \begin{aligned}
            \hat{\beta} & = \frac{\sum_{1}^{N} x_{i} y_{i}}{\sum_{1}^{N} x_{i}^{2}}   \\
            r_{i} & = y_{i} - x_{i} \hat{\beta}
          \end{aligned}
        $$                                                             {#eq:3.24}
       Using vector notation, where ${\bf y} = (y_{1}, \ldots, y_{N})^{T}$ and  
       ${\bf x} = (x_{1}, \ldots, x_{N})^{T}$
        $$
          \begin{aligned}
            \langle {\bf x}, {\bf y} \rangle & = \sum_{i=1}^{N} x_{i} y_{i}     \\
                & = {\bf x}^{T} {\bf y}
          \end{aligned}
        $$                                                             {#eq:3.25}
       Rewriting eqn \ref{eq:3.24} in vector notation
        $$
          \begin{aligned}
            \hat{\beta} & = \frac{\langle {\bf x}, {\bf y} \rangle}{\langle {\bf x}, {\bf x} \rangle}\\
            {\bf r} & = {\bf y} - {\bf x} \hat{\beta}
          \end{aligned}
        $$                                                             {#eq:3.26}
    #) Now consider that that there are multiple variables measured (i.e. $p > 1$) and 
       that each is independent (i.e. orthoganol). 
        #. Inputs (columns of matrix ${\bf X}$), ${\bf x}_{1}, \ldots, {\bf x}_{p}$
        #. i.e. $\langle {\bf x}_{j}, {\bf x}_{k} \rangle = 0$, for all $j \ne k$
    #) Now since each variable is independent, the multiple least squares estimate for each 
       variable is simple the univariate estimates
        $$
          \begin{aligned}
            \hat{\beta}_{j} & = \frac{\langle {\bf x}_{j}, {\bf y} \rangle}{\langle {\bf x}_{j}, {\bf x}_{j} \rangle}\\
          \end{aligned}
        $$                                                            
        #. Orthogonal inputs (variables), occur most often with balanced, well-designed
           experiments
        #. Never occurs with observational data
            * Can help situation by orthogonalizing them
    #) NOW consider that we have an intercept and a single input / variable ${\bf x}$
        $$
          \begin{aligned}
            \hat{\beta}_{1} & = \frac{\langle {\bf x} - \overline{x}{\bf 1}, {\bf y} \rangle}{\langle {\bf x} - \overline{x}{\bf 1}, {\bf x} - \overline{x}{\bf 1} \rangle}\\
          \end{aligned}
        $$                                                             {#eq:3.27}
       where $\overline{x} = \sum_{i} x_{i} / N$ and ${\bf 1} = 1_{1}...1_{N} = {\bf x}_{0}$. 
    #) You can get eqn \ref{eq:3.27} by doing two applications of regression with
       eqn \ref{eq:3.26}
        #. Regress ${\bf x}$ on ${\bf 1}$. Produces residual
           ${\bf z} = {\bf x} - \overline{x} {\bf 1}$
            * "Orthogalizes" ${\bf x}$ w/r/t ${\bf 1}$
        #. Regress ${\bf y}$ on the residual ${\bf z}$ to give the coefficient
           $\hat{\beta}_{1}$
            * Simple univariate regression using orthogonal predictors ${\bf 1}$ and
              ${\bf z}$
    #) Jargon
        #. "regress ${\bf b}$ on ${\bf a}$" means a simple univariate regression without 
           intercept
            * Coefficient : $\hat{\gamma} =
              \langle {\bf a}, {\bf b} \rangle / \langle {\bf a}, {\bf a} \rangle$
            * Residual    : ${\bf b} - \hat{\gamma} {\bf a}$
        #. "${\bf b}$ is orthogonalized w/r/t ${\bf a}$"
    #) This recipe can be generalized to $p$ inputs
    #) Fig. 3.4 :
        #. I understand that ${\bf z}$ as the residual and is perpendicular to
            ${\bf x}_{1}$
            * Makes sense b/c if ${\bf x}_{1}$ and ${\bf x}_{2}$ were the same 
              vector, there would be no residual and ${\bf z} = 0$
        #. If you want to REMOVE the dependency of ${\bf x}_{2}$ on ${\bf x}_{1}$ the
           residual will be the DIFFERENCE between ${\bf x}_{2}$ on ${\bf x}_{1}$. So
           If you regress on the residual, the difference is removed.
        #. This figure is confusing b/c it is really showing two steps, he should show the
           residual step separate from the ${\bf \hat{y}}$
    #) Algorithm 3.1 : Regression by Successive Othogonalization
        #. Step 1 : Initialize ${\bf z}_{0} = {\bf x}_{0} = {\bf 1}$
        #. Step 2 : For $j = 1, 2, \ldots, p$
            * Regress $x_{j}$ on ${\bf z}_{0}, {\bf z}_{1}, \ldots, {\bf z}_{j-1}$ to
              produce coeffs $\hat{\gamma} = \langle {\bf z}_{l}, {\bf x}_{j} \rangle / \langle {\bf z}_{l},{\bf z}_{l} \rangle$
              where $l = 0, \ldots, j-1$ and the residual vector is 
              ${\bf z}_{j} = {\bf x}_{j} = \sum_{k=0}^{j-1} \hat{\gamma}_{kj} {\bf z}_{k}$
        #. Step 3 : Regress ${\bf y}$ on the residual ${\bf z}_{p}$ to give estimate $\hat{\beta}_{p}$
        $$
          \begin{aligned}
            \hat{\beta}_{p} & = \frac{\langle {\bf z}_{p}, {\bf y} \rangle}{\langle {\bf z}_{p}, {\bf z}_{p} \rangle}\\
          \end{aligned}
        $$                                                             {#eq:3.28}
            * Note that eqn \ref{eq:3.28} is for the LAST variable $p$
            * ${\bf x}_{j}$ is a linear combination of ${\bf z}_{k}$, for $k \le j$
            * ${\bf x}_{j}$ are the column vectors of ${\bf X}$, i.e. the variables
            * ${\bf z}_{j}$ are all orthogonal, forme basis for ${\bf X}$
            * Eqn \ref{eq:3.28} is key Correlated inputs in multiple regression
            * Can shuffle ${\bf x}_{j}$, eqn \ref{eq:3.28} still holds
            * Shown that $j$th multiple regression coefficient is the : 
                + univariate regression of ${\bf y}$ on
                  ${\bf x}_{j \cdot 012 \ldots (j-1)(j+1) \ldots, p}$
                + the residual after regressing ${\bf x}_{j}$ on ${\bf x}_{0}, {\bf x}_{1},
                  \ldots, {\bf x}_{j-1}, {\bf x}_{j+1}, \ldots, {\bf x}_{p}$
                + QUESTION : Let's discuss this
            * \emph{The multiple regression coefficient $\hat{\beta}_{j}$ represents
              the additional contribution of ${\bf x}_{j}$ on ${\bf y}$, after 
              ${\bf x}_{j}$ has been adjusted for ${\bf x}_{0}$, ${\bf x}_{1}$, \ldots,
              ${\bf x}_{j-1}$, ${\bf x}_{j+1}$, $\ldots$, ${\bf x}_{p}$}
                + It makes intuitive sense that you want to remove the intervariable 
                  dependence before making a prediction with ${\bf \hat{\beta}}$
            * If ${\bf x}_{p}$, is highly correlated with some of the other ${\bf x}_{k}$
              residual vector ${\bf z}_{p} \approx 0$
                + $\hat{\beta}_{p}$ will be unstable
                + True for other variables
                + In this situation, all the Z-scores might be small. Anyone can
                  be deleted, but not all. 
    #) Alternate formula for variance estimates (recall eqn \ref{eq:3.8})
        $$
          \begin{aligned}
            \text{Var}(\hat{\beta}_{p}) = \frac{\sigma^{2}}{\langle {\bf z}_{p} {\bf z}_{p} \rangle}
          \end{aligned}
        $$                                                             {#eq:3.29}
        #. In words, can estimate $\hat{\beta}_{p}$ based on length of residual vector 
           ${\bf z}_{p}$; it represents how much of ${\bf x}_{p}$ is unexplained by other
           ${\bf x}_{k}$'s 
        #. See Exercise 3.4 for Gram-Schmidt procedure
        #. Intuition here is that if $|{\bf z}_{p}|$ is small (ie they are highly 
           correlated), then of course you won't get the separation you need.
        #. QUESTION : What is $\sigma$ here, should it be $\sigma_{p}$?
        #. ANSWER   : $\sigma$ is a variance also. From near eqn \ref{eq:3.8} (eqn below 
                      \ref{eq:3.8} in my notes) it is the variance observations, ${\bf y}$.
                      Observed variance
    #) Reformulate Algorithm 3.1 in matrix form 
        #. Step 2
        $$
          \begin{aligned}
            {\bf X} = {\bf Z}{\bf \Gamma}
          \end{aligned}
        $$                                                             {#eq:3.30}
           where : 
            * ${\bf Z}$ has the columns of ${\bf z}_{j}$ (in order)
            * ${\bf \Gamma}$ is upper triangular matrix with entries $\hat{\gamma}_{kj}$
        #. Introduce Diagonal matrix ${\bf D}$ with $j$th diagonal entry
           $D_{jj} = ||z_{j}||$
        $$
          \begin{aligned}
            {\bf X} & = {\bf Z}{\bf D}^{-1}{\bf D}{\bf \Gamma}          \\
            {\bf X} & = {\bf Q}{\bf R}                                  \\
          \end{aligned}
        $$                                                             {#eq:3.31}
            * ${\bf D}$ is invertible
            * Enter $QR$ decomposition
            * ${\bf Q}$ is an $N \times (p + 1)$ orthogonal matrix  
                + ${\bf Q}^{T}{\bf Q} = {\bf I}$
            * ${\bf R}$ is a $(p + 1) \times (p + 1)$ upper triangular matrix  
        $$
          \begin{aligned}
            \hat{\beta} = {\bf R}^{-1} {\bf Q}^{T} {\bf y}
          \end{aligned}
        $$                                                             {#eq:3.32}
        $$
          \begin{aligned}
            \hat{y} = {\bf Q} {\bf Q}^{T} {\bf y}
          \end{aligned}
        $$                                                             {#eq:3.33}
            * eqn \ref{eq:3.33} is easy to solve, see Exercise 3.4
#. 3.2.4 - Multiple Outputs
    a) Now consider multiple outputs $Y_{1}, Y_{2}, \ldots, Y_{K}$ from multiple inputs
       $X_{0}, X_{1}, \ldots, X_{p}$
        #. So far we've only considered multiple inputs with 1 output, 
        #. There are $p$ variables
        $$
          \begin{aligned}
            Y_{k} & = \beta_{0k} \sum_{j=1}^{p} X_{j} \beta_{jk} + \epsilon_{k} \\
          \end{aligned}
        $$                                                             {#eq:3.34}
        $$
          \begin{aligned}
            Y_{k} & = f_{k}(X) + \epsilon_{k}                                   \\
          \end{aligned}
        $$                                                             {#eq:3.35}
       With $N$ training cases, the model in matrix notation is 
        $$
          \begin{aligned}
            {\bf Y} & = {\bf X}{\bf B} + {\bf E}                    \\
            & \text{where : }                                       \\
            & \text{ ${\bf Y}$ is $N \times K$ response matrix with $ik$ entry $y_{ik}$} \\
            & \text{ ${\bf X}$ is $N \times (p+1)$ input matrix } \\
            & \text{ ${\bf B}$ is $(p+1) \times K$ parameter matrix (like $\beta$ for multiple outputs} \\
            & \text{ ${\bf E}$ is $N \times K$ error matrix} \\
          \end{aligned}
        $$                                                             {#eq:3.36}
       Generalizing eqn \ref{eq:3.2}
        $$
          \begin{aligned}
            \text{RSS}({\bf B}) & = \sum_{k=1}^{K} \sum_{i=1}^{N} (y_{ik} - f_{k}(x_{i}))^{2} \\
          \end{aligned}
        $$                                                             {#eq:3.37}
        $$
          \begin{aligned}
            \text{RSS}({\bf B}) & = \text{tr}[({\bf Y} - {\bf X}{\bf B})^{T}({\bf Y} - {\bf X}{\bf B} \\
          \end{aligned}
        $$                                                             {#eq:3.38}
       Note ${\bf B}$ is the equivalent to ${\beta}$. Analoguosly, minimizing the residuals
        $$
          \begin{aligned}
            \hat{\bf B} & = ({\bf X}^{T}{\bf X})^{-1}{\bf X}^{T}{\bf Y}
          \end{aligned}
        $$                                                             {#eq:3.39}
       The coefficients for $k$th outcome is the least squares regression of ${\bf y}_{k}$ on
       ${\bf x}_{0}, {\bf x}_{1}, \dots, {\bf x}_{p}$
        #. Multiple outputs don't affect the individual ${\bf y}_{k}$
        #. QUESTION : Is that always true?

       If the errors $(\epsilon = \epsilon_{1}, \ldots, \epsilon_{K}$ in \ref{eq:3.34}
       correlated, let $\text{Cov}(\epsilon) = {\bf \Sigma}$, eqn \ref{eq:3.37} becomes 
        $$
          \begin{aligned}
            \text{RSS}{\bf B; \Sigma} & = \sum_{i=1}^{N}(y_{i} - f(x_{i}))^{T}\Sigma^{-1}(y_{i} - f(x_{i}))     \\
            & \text{where : }                                       \\
            & \text{$f(x) = (f_{1}(x), \ldots, f_{K}(x))^{T}$}      \\
            & \text{$y_{i} is vector of $K$ responses for observation $i$ (of $N$)}     \\
          \end{aligned}
        $$                                                             {#eq:3.40}
        #. QUESTION : How do you derive eqn \ref{eq:3.40}?  Looks like an expectation value
        #. ANSWER : probably not too far off.
       Despite eqn \ref{eq:3.40}, eqn \ref{eq:3.39} is still a solution to eqn \ref{eq:3.40}.
       If ${\bf \Sigma}_{i}$ varies among observations, then you are screwed and ${\bf B}$
       becomes more complex.
    #) N : The big takeaway is that this is what you'd expect.
        #. Ali : Not surpising, but pleasing when it works out to what you'd make
                 an assumption on 
        #. Ali : The epsilon's are for EACH output.
    #) N : Back to House case (outputs = price, time on market) it makes sense
        #. Ali : But I'd expect an interaction term, I don't see that here.
        #. N : Of course that would be harder
        #. A : Actually this is eqn \ref{eq:3.40},  this is why he uses the
               $\text{Cov}(\epsilon)$
            * But this is only the ERRORS, but maybe that is all you know in an
              underdermined system
            * We're not sure if that is on the 
            * Maybe he's taking an observational perspective
        #. N : Is it that the errors are correlated
        #. N : What is ${\bf \Sigma}_{i}$?
            * A : Per sample? But how do you have a covariance matrix with 1 sample, maybe
                  it is 1 row in ${\bf \Sigma}$?  That'd make more sense.
     

    #) Q / A
        #. Q : Below eqn \ref{eq:3.26}, he uses bold for the measurements ${\bf x}_{1}$.
               In the current case $x_{1}$ should be a scalar
        #. A : No, it is ${\bf x}_{p}$. For variable $p$, there are multiple entries for each
               measurement of that variable. 
        #. Q : The 'intercept' is what he previously called the \emph{bias}, right?
        #. A : No...
            * Ali : Look at p11, between eqn 2.1 and 2.2. $\hat{\beta}_{0}$ is called the 
                    \emph{bias}
            * Maybe nomenclature? Maybe it means two things that aren't quite the same thing.
        #. Q : Eqn \ref{eq:3.27} is called 'centering', I don't see how he pulls 
               it from the ether? I don't think you get that if you pulled. The 
               intercept is not the mean.
        #. A : Maybe you should derive it yourself. "Have faith or just prove it"
        #. Q : Eqn \ref{eq:3.27} smells like the Gram-Schmidt process
        #. A : It is, just he's being awkward.
        #. Q : Why are we regressing against ${\bf 1}$
        #. A : He's just picking the ${\bf 1}$ as the starting point of his Gram-Schmidt
               process. $f(x)=\beta_{0}*{\bf 1} + \sum {\bf x}_{i} \beta_{i}$
        #. Q : Algorithm 3.1 seems a lot like a bootstrapping method
        #. A : Need previous z's and current x to make the next z. You have two 
               vectors, the dot products get's the projection.  If you subtract the 
               projection, you get the normal component
        #. Q : Why do multiple regression with successive othrogonalization vs 
               just normal linear regression? 
        #. A : If you have orthoganol inputs, your $\beta_{j}$'s are cleaner.
        #. Q : in eqn \ref{eq:3.28}, $\beta_{p}$ only refers to a single vector, right? 
        #. A : $\hat{\beta}_{p}$ is a scalar. The definition of a an inner product has to 
               map into a scalar. Inner products are NOT matrix multiplication.
    #) Meeting - 27jan2023
        #. It is hard to find an example where the $Y_{0}, Y_{1}, ..., Y_{p}$
            * E.g. ability ice skating and GPA...but those are correlated
              
3.3 Subset Selection
==========================
#. Intro
    a) 2 Reaons we aren't satisfied with least squares estimates (eqn \ref{eq:3.6})
        #. Prediction accuracy 
            * tend to have low bias, but large variance.
            * Can shrink  some coeffs and sacrifice a little bias to reduce th variance
        #. Intepretation
            * with large number of predictors, should subset to find ones with
              strongest effects
#. 3.3.1 : Best-Subset Selection
    a) Best subset of $k \in \{0,1,\ldots,p\}$ 
        #. Gives smallest residual sum of squares in eqn \ref{eq:3.2}
        #. \emp{leaps and bounds} algorithm
            * see : Furnival and Wilson, 1974
            * feasible for $p$ as large as 30 or 40.
    #) Fig 3.5
        #. Shows all subset models for prostate cancer data.
        #. Size 2 gets you most of the way, size 8 is marginally better
    #) Use cross-validation to estimate prediction error to get $k$
        #. AIC criterion is popular
#. 3.3.2 : Forward and Backward-Stepwise Selection
    a) Need method for selecting subset
        #. Infeasible to sample all possible subsets for $p$ much larger than 40 
        #. \emph{Forward-stepwise selection} 
            * Starts with intecept and sequentially adds to into the model the
              predictor that most improves the fit
            * \emph{greedy algorithm}
            * Reasons to prefer :
                + \emph{Computational} : for large $p$ we cannot compute the best subset
                                         sequence, but we can always compute the forward
                                         stpwise sequence (even when $p >> N$)
                + \emph{Statistical} : a price is paid in variance for selecting the best
                                       subset of each size; forward stepwise is a more 
                                       constrained search and will have lower variance
                                       and perhaps more bias
    #) Figure 3.6
        #. Shows several subset selection techniques
            * Best Subset
            * Forward Stepwise
                + Can be used either when $N>p$ or $N<p$
            * Backward Stepwise
                + starts with full model, sequentially delete predictor and has least 
                  impact on fit.
                + Drop candidate with smallest Z-score
                + Can only be used when $N>p$
            * Forward Stagewise
                + 
        #. Simulated lin reg problem 
            * $Y = X^{T} \beta + \epsilon$
            * $N = 300$ observations 
            * $p=31$ Gaussian variables
            * Pairwise correlations = 0.85
                + QUESTION : What is this?
            * Coefficients for 10 vaiables drawn at random from $G(0,0.4)$
            * Noise $\epsilon \sim N(0, 6.25)$
            * Results averaged over 50 simulations
                + QUESTION : Same 10 variables for each simulation?
        #. Observations
            * Best subset, Forward Stepwise  and Backward Stepwise all performed about the same
    #) Prostate cancer revisited
        #. best-subset, forward and backward selection ALL gave same sequence of terms
    #) Software
        #. R - step package uses AIC criterion to weigh choices
        #. Other packages use F-statistics
            * Add "significant" and drop "non-significant" terms
        #. Tempting to print model summary bug 
            * WARNING : standard errors don't account for search process
#. 3.3.3 Forward-Stagewise Regression
    a) Procedure
        #. Initial intercept = $\bar{y}$, centered predictors = 0
        #. At each step algorithm identifies the variable most correlated with current 
           residual
        #. On this variable computes simple linear regression coefficients
        #. Add this to current coefficient for that variable
        #. Continue until none of the variables have correlation with the residuals.
            * i.e. least squares fit when $N>p$
            * QUESTION : discuss procedure
    #) Comment
        #. more constrained than forward-stepwise regression
        #. Unlike forward-stepwise regression, none of the vars are adjusted when a
           term is added to the model
            * Traditionally, discarded as inefficient on converge on least squares fit
            * Has advantages in high-dimensional problems compared to the others
    #) Big difference between Forward-Stagewise and Forward-Stepwise
        #. Stepwise adds variables that. Once and done
        #. Stagewise keeps all variables and may modify multiple times, which is why
           it takes longer to converge
            * At each step the algorithm identifies the variable most correlated
              with the current residual.
            * Hard sentenct to understand, Ali thinks it means that the variable that
              best fits the current globabl fit. 
#. 3.3.4 Prostate Cancer Data Example (continued)
    a) Table 3.3
        #. Estimated coefficients and test error results for different subset and shrinkage
           methods applied to the prostate data. The blank entries corrspond to variables 
           omitted
        #. Tested 
            * best-subset selection
            * ridge regression
            * lasso
            * principal components regression
            * partial least squares
        #. Used cross-validation
            * Train on 90% of data, test on 10%
    #) Figure 3.7
        #. Use \emph{one-standard-error} rule
            * QUESTION : Discuss!
            * ANSWER   : Standard deviation of the population is an ideal,
                         Standard deviation of the sample
        #. Shows number of directions (p?) vs Cross-validation error
            * QUESTION : Let's discuss this plot
            * ANSWER   : 

3.4 Subset Selection
==========================
1. Ridge Regression 
    a) Shrinks regression coefficients by imposing penalty on size
        $$
          \begin{aligned}
            \hat{\beta}^{ridge} = \text{argmin}\Big{\sum_{i=1}^{N}(y_{i} - \beta_{0} - \sum_{j=1}^{p} x_{ij} \beta_{j})^{2} + \lambda \sum_{j=1}^{p} \beta_{j}^{2}\Big}
          \end{aligned}
        $$                                                             {#eq:3.41}
       where $\lambda \ge 0$ is a complexity param that controls the amount of shrinkage.
        #. Larger $\lambda$, larger shrinkage
        #. Idea of penalyzing by sum-of-squares is used in neural networks and is known as
           \emph{weight decay}
    #) Equivalent way to write ridge regression
        $$
          \begin{aligned}
            \hat{\beta}^{ridge} & = \text{argmin}_{\beta}\Big{\sum_{i=1}^{N}(y_{i} - \beta_{0} - \sum_{j=1}^{p} x_{ij} \beta_{j})^{2} \Big}   \\
                                & \text{subject to } \quad \sum_{j=1}^{p} \beta_{j}^{2} \leq t
          \end{aligned}
        $$                                                             {#eq:3.42}
        #. QUESTION : Is $\hat{\beta}^{ridge}$ a vector or not? eq 2.6 has it as a vector (p12)
        #. ANSWER : Yes, see \ref{eq:3.44}
        #. One-to-one correspondance between $\lambda$ and $t$ in \ref{eq:3.41} and
           \ref{eq:3.42} respectively
        #. When there are many correlated variables in linear regression model, coeffs 
           can become poorly determined and have high variance
            * E.g. a wildly large coeff by cancelled by its equally negative correlated cousin
            * \ref{eq:3.42} alleviates that problem
        #. Must standardize inputs before solving \ref{eq:3.41}
        #. Note that intercept, $\beta_{0}$ was left out of penalty term
            * If we penalized the intercept, it would make the procedure depend on the 
              origin chose for $Y$
            * i.e. adding a constant $c$ wouldn't just `simply' shift the predictions
              (which is bad)
            * Procedure
                + Estimate $\beta_{0}$ by $\overline{y} = \frac{1}{N} \sum_{1}^{N} y_{i}$
                + Estimate rest of coeffs via ridge regression w/o intercept using centered
                  $x_{ij}$ (i.e. $x_{ij} - \overline{x}_{j}$)
            * From now on assume that the centering has already occurred
            * QUESTION : Why hasn't he standardized the variance as well?
        #. Writing \ref{eq:3.42} in matrix form
        $$
          \begin{aligned}
            \text{RSS}(\lambda) = ({\bf y} - {\bf X}\beta)^{T}({\bf y} - {\bf X}\beta) + \lambda\beta^{T}\beta
          \end{aligned}
        $$                                                             {#eq:3.43}
           re-arranged as : 
        $$
          \begin{aligned}
            \hat{\beta}^{ridge} = ({\bf X}^{T}{\bf X} + \lambda {\bf I})^{-1}{\bf X}^{T}{\bf y}
          \end{aligned}
        $$                                                             {#eq:3.44}
           where ${\bf I}$ is $p \times p$ identity matrix. The addition to the diagonals
           ensures that the solution is non-singular.
        #. For othonormal inputs, ridge estimates are just scaled version of least squares,
           $\hat{\beta}^{ridge} = \hat{\beta}/(1+\lambda)$
        #. QUESTION : What is the y-axis of Figure 3.8?
        #. QUESTION : I don't understand the log-posterior discussion on the bottom of p64
    #) Singular value decomposition (SVD) on centered input matrix ${\bf X}$
        #. Give additional insight into nature of ridge regression, for ${\bf X}$ an
           $N \times p$ matrix :
        $$
          \begin{aligned}
            {\bf X} = {\bf U}{\bf D}{\bf V}^{T}
          \end{aligned}
        $$                                                             {#eq:3.45}
            * ${\bf U}$ is $N\times p$ orthogonal matrix w/ columns of ${\bf U}$
              spanning column space of ${\bf X}$ 
            * ${\bf V}$ is a $p \times p$ orthogonal matrix w/ columns of ${\bf V}$
              spanning rows of ${\bf X}$
            * ${\bf D}$ is a $p \times p$ diagonal matrix of the singular values of
              ${\bf X}$
        #. Using SVD, can write least squares fit vector as 
        $$
          \begin{aligned}
            {\bf X}\hat{\beta}^{ls} & = {\bf X}({\bf X}^{T}{\bf X})^{-1}{\bf X}^{T}{\bf y} \\
                                    & = {\bf U}{\bf U}^{T}{\bf y} \\
          \end{aligned}
        $$                                                             {#eq:3.46}
           where ${\bf U}^{T}{\bf Y}$ are the coordinates of ${\bf y}$ w/r/t orthonormal
           basis ${\bf U}$. Similar to eq 3.33. Now the ridge regression : 
        $$
          \begin{aligned}
            {\bf X}\hat{\beta}^{ridge} & = {\bf X}({\bf X}^{T}{\bf X} + \lambda{\bf I})^{-1}{\bf X}^{T}{\bf y} \\
                & = {\bf U}{\bf D}({\bf D}^{2} + \lambda{\bf I})^{-1}{\bf D}{\bf U}^{T}{\bf y} \\
                & = \sum_{j=1}^{p} {\bf u}_{j} \frac{d_{j}^{2}}{d_{j}^{2} + \lambda}{\bf u}_{j}^{T}{\bf y}
          \end{aligned}
        $$                                                             {#eq:3.47}
           where ${\bf u}_{j} are columns of ${\bf U}$ and $d_{j}$ are the diagonal elements
           off of ${\bf D}$
        #. SVD is another way of expressing \emph{principal components} of the variables
           in ${\bf X}$
            * Sample covariance matrix is : ${\bf S} = {\bf X}^{T} {\bf X} / N$, and from 
              \ref{eq:3.45} we get the \emph{eigen decomposition} of ${\bf X}^{T}{\bf X}$
              and of ${\bf S}$, up to a factor $N$).
        $$
          \begin{aligned}
            {\bf X}^{T}{\bf X} = {\bf V}{\bf D}^{2}{\bf V}^{T}
          \end{aligned}
        $$                                                             {#eq:3.48}
              columns of ${\bf V}$ are the eigenvectors, aka principal components 
              directions of ${\bf X}$. 
            * First Princp Comp is the eigenvecot s.t.
              ${\bf z}_{1} = {\bf X}v_{1} = {\bf u}_{1} d_{1}$ 
              has the largest sample variance amongst all normalized linear combos, see :`
        $$
          \begin{aligned}
            Var({\bf z}_{1} = Var(\bf{X} v_{1}) = d_{1}^{2} / N
          \end{aligned}
        $$                                                             {#eq:3.49}
        #. QUESTION : I don't see how eqn \ref{eq:3.47} and ridge regression is related
                      to principal components
        #. QUESTION : Let's discuss Figure 3.9
        #. In Fig 3.7, plotted the estimated prediction error vs \emph{effective degrees of fredom} : 
        $$
          \begin{aligned}
            df(\lambda) & = \tr[{\bf X}({\bf X}^{T}{\bf X} + \lambda {\bf I})^{-1}{\bf X}^{T}] \\
                & = \tr[{\bf H}]    \\
                & = \sum_{j=1}^{p} \frac{d_{j}^{2}}{d_{j}^{2} + \lambda}
          \end{aligned}
        $$                                                             {#eq:3.50}
           Note : degrees of freedom : 
            * $\text{df}(\lambda) = p$, when $\lambda = 0$
            * $\text{df}(\lambda) = 0$, when $\lambda = \inf$
            * don't forget about additional dof b/c of intercept
            
#.
#.  
            

Exercises
==========================
1. Exercises that I think would be worth doing:

To Do
==========================
1. Work on derivation of eqn \ref{eq:3.8}
#. Add code to illustrate normal vs student's t-distribution
#. Try to implement Algorithm 3.1
#. 3/10/23 : He got to eq : 3.45
    a) 40k classified groups, use nic's code?
#. Try to reproduce Table 3.3

