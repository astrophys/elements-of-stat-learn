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
            F & = \frac{(\text{RSS}_{0} - \text{RSS}_{1})/(p_{1} - p_{0}}{\text{RSS}_{1}/(N-p_{1}-1} \\
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
    #) CI for entire vector $\beta$
        $$
          \begin{aligned}
            C_{\beta} = \{\beta|(\hat{\beta}-\beta)^{T}{\bf X}^{T}{\bf X}(\hat{\beta} - \beta) \leq \hat{\sigma}^{2} \chi_{p+1}^{2(1-\alpha)}\}
          \end{aligned}
        $$                                                                      {#eq:3.15}
        #. QUESTION : Typo for $\chi$ in eqn \ref{eq:3.15}? Explain?
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



3.3 Two Simple Approaches to Prediction, Least Squares and Nearest Neighbors
==========================
#. QUESTIONS : 

Exercises
==========================
1. Exercises that I think would be worth doing:

To Do
==========================
1. Work on derivation of eqn \ref{eq:3.8}
#. Add code to illustrate normal vs student's t-distribution

