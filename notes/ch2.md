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
        $$ \frac{d}{d \beta} \big( RSS(\beta) = ({\bf y} - {\bf X} \beta)^{T}({\bf y} - {\bf X}\beta)\big)$$ }
        $$ \frac{d}{d \beta} \big( RSS(\beta) = ({\bf y}^{T} - \beta^{T}{\bf X}^{T} )({\bf y} - {\bf X}\beta)\big)$$ 
        $$ \frac{d}{d \beta} \big( RSS(\beta) = {\bf y^{T}y} -{\bf y^{T}X\beta} - \beta^{T}{\bf X}^{T}{\bf y} + \beta^{T}{\bf X}^{T}{\bf X}\beta \big)$$
        $$  0 = - {\bf y^{T}X} - {\bf X}^{T}{\bf y} + \beta^{T}{\bf X}^{T}{\bf X}\beta \big)$$


2.4 Statistical Decision Theory
==========================

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
#. 
