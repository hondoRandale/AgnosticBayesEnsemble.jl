#Agnostic Bayes Ensemble Documentation 
 
__Overview__

This package has been developed to facilitate increased predictive performance, by combining raw base models in an agnostic fashion, 
i.e. the methods don’t use any assumption regarding the used raw models. Furthermore, we specifically implemented ensemble algorithms 
that can deal with arbitrary loss function and with regression and classification problems, this holds true for all, except for the
 dirichletPosteriorEstimation algorithm, which is limited to classification problems.
 
 **Hint**: In most cases it is advisable to _deactivate_ Hyperthreading for best performance.
However, in some rare cases – depending on the (hardware) platform the package runs on- you
will get the best performance with Hyperthreading enabled, to be sure, it is best practice to
measure the performance with and without Hyperthreading.
 
## List of Algorithms
 
```@docs
bootstrapPosteriorEstimation
```
 
 ```@docs
bootstrapPosteriorEstimation!
```
 
```@docs
bootstrapPosteriorCorEstimation
```

```@docs
dirichletPosteriorEstimation
```

```@docs
dirichletPosteriorEstimationV2
```

```@docs
GMatrix
```

```@docs
dirichletPosteriorEstimation!
```

```@docs
metaParamSearchValidationDirichlet
```



## Tutorials
```@contents
Pages = [
    "tutorials/page1.md",
    "tutorials/page2.md",
    "tutorials/page3.md"
    ]
Depth = 2
```
 
## Another Section
```@contents
Pages = [
    "sec2/page1.md",
    "sec2/page2.md",
    "sec2/page3.md"
    ]
Depth = 2
```
 
## Index
 
```@index
```