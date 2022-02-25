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
        #. transformations of quantitative inputs (e.g. log, square= =-root)
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
       the `hat' matrix b/c it puts a `hat' on ${\bf y}$. Aka the `projection matrix'
        #. QUESTION : Shouldn't eqn \ref{eq:3.7} have ${\bf X}^{T}$. See text below
           eqn 2.6?
    #) Figure 3.2     
        #. Geometrical representation of least squares estimate
        #. Denote column vectors ${\bf X}$ as ${\bf x}_{0}, {\bf x}_{1}, \ldots, {\bf x}_{p}$
           with ${\bf x}_{0} = 1$
    #) Minimize $\text{RSS}(\beta)$ by choosing $\hat{\beta}$ so that residual vector 
       ${\bf y} - {\bf \hat{y}}$ is \emph{orthogonal} to this subspace
        #. QUESTION : Explain this, I don't have intuition on why it is 'orthoganol'
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
        #. $x_{i}$ are fixed (i.e. non-random)
    #) Variance-covariance mastrix of least squres params $(\beta)$ is derived from
       eqn \ref{3.6}.
       Starting with eqn \ref{3.6}.
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
       is fixed, so only the ${\bf y}$ component of eqn \ref{3.6} can vary. 
        $$
          \begin{aligned}
            \text{Var}(\hat{\beta}) & = \text{E}[(\hat{\beta} - \overline{\beta})]^{2} \\
            & \vdots \\
            & \text{and by some not-so-easy magic}    \\
            & \vdots \\
            \text{Var}(\hat{\beta}) & = ({\bf X}^{T}{\bf X}
          \end{aligned}
        $$
       
        #. QUESTION : Is my derivation correct? What is exactly $\overline{\beta}$. Is it 
           $\text{mean}(\hat{\beta})$ or is it $\text{mean}(\hat{\beta_{i}})$ ?


3.3 Two Simple Approaches to Prediction, Least Squares and Nearest Neighbors
==========================
#. QUESTIONS : 

Exercises
==========================
1. Exercises that I think would be worth doing:
