

**AgnosticBayesEnsemble**  





​           __Auticon Analytics__



![Auticon_Grafik](C:\Users\Pc-Lap183\Desktop\JuliaPackages\production\AgnosticBayesEnsemble\Auticon_Grafik.PNG)

**_Overview_**

This package has been developed to facilitate increased predictive performance, by combining raw base models in an agnostic fashion, i.e. the methods don’t use any assumption regarding the used raw models. Furthermore, we specifically implemented ensemble algorithms that can deal with arbitrary loss function and with regression and classification problems, this holds true for all, except for the dirichletPosteriorEstimation algorithm, which is limited to classification problems.

**Hint**: In most cases it is <u>advisable</u> to deactivate Hyperthreading for best performance.
However, in some rare cases – depending on the (hardware) platform the package runs on- you will get the best performance with Hyperthreading enabled, to be sure, it is best practice to measure the performance with and without Hyperthreading.



**_low level Interface_**



The Interface was designed to be easy to use, therefore all parameters needed by the algorithms in the package are either y_1, y_2, y_3, …, y_k the predictions per raw model along with the label vector T,
Or alternatively e_1, e_2, e_3, …, e_k the error between predicted and real labels and ground truth T.
Some of the methods need additional (prior-) parameters, however this simple basic structure is consistent along all implemented ensemble methods in this package.



 `*## @param: errMat         Matrix  - each column is prediction of one hypothesis*`

 `*## @param: samplingFactor Float64 - relative samples taken per iteration*` 

 `*## @param: nrRuns         Int64   - number of iterations over entire set*`

 `*## @brief: compute posterior p( h\* = h | S )*`

 `*## @return: posterior Float{Float64,1} - Distribution p( h\* = h | S )*`

 `function bootstrapPosteriorEstimation( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )`

___

 `*## @param: errMat Matrix  - each column is prediction of one hypothesis*` 

 `*## @param: nrRuns Int64   - number of sampling runs*`

 `*## @param: α_     Float64 - scalar prior parameter*`

 `*## @brief: compute posterior p( h\* = h | S )*`

 `*## @return: posterior   Float{Float64,1} - Distribution p( h\* = h | S )*`

 `function dirichletPosteriorEstimation( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64 )`

___

`## @param: predictions     Matrix{Float64}  - each column is prediction of one hypothesis`

`*## @param: y              Array{Float64,1} - ground truth label vector*`

`*## @param: samplingFactor Float64          - relative samples taken per iteration*` 

`*## @param: nrRuns         Int64            - number of iterations over entire set*`

`*## @brief: compute posterior p( h\* = h | S )*`

`*## @return: posterior   Float{Float64,1} - Distribution p( h\* = h | S )*`

`function bootstrapPosteriorCorEstimation( predictions::Matrix{Float64}, y::Array{Float64,1}, samplingBatchSize::Int64, nrRuns::Int64 )`

___

 `*## @param: predictions Matrix - each column is prediction of one hypothesis*` 

 `*## @param: nrRuns      Int64  - number of sampling runs*`

 `*## @brief: compute posterior p( h\* = h | S )*`

 `*## @return: posterior   Float{Float64,1} - Distribution p( h\* = h | S )*`

 `function TDistPosteriorEstimation( errMat::Matrix{Float64}, nrRuns::Int64 )`

___

**_Examples_**



`using AgnosticBayesEnsemble`

`using DataFrames`

`using Random`

`using Statistics`

`using StaticArrays`

`using Optim`

`using MultivariateStats`



#== create artificial predictions and ground truth ==#

`function distortBinaryPrediction( y::BitArray{1}, distortionFactor::Float64 )`

​    `res     = deepcopy( y );`  

​    `indices   = rand( 1:1:size( y, 1 ), round( Int64, distortionFactor * size( y, 1 ) ) );`

​    `res[indices] = .!y[indices];`

​    `return res;`

`end`  

`n  = 100000;`

`y  = Bool.( rand( 0:1,n ) );`

`yH1 = distortBinaryPrediction( y, 0.20 );`

`yH2 = distortBinaryPrediction( y, 0.21 );`

`yH3 = distortBinaryPrediction( y, 0.22 );`

`yH4 = distortBinaryPrediction( y, 0.23 );`

`yH5 = distortBinaryPrediction( y, 0.24 );`

`yH6 = distortBinaryPrediction( y, 0.24 );`

`yH7 = distortBinaryPrediction( y, 0.26 );`

`yH8 = distortBinaryPrediction( y, 0.27 );`

`yH9 = distortBinaryPrediction( y, 0.28 );`

`yH10 = distortBinaryPrediction( y, 0.29 );`

`yH11 = distortBinaryPrediction( y, 0.30 );`

`yH12 = distortBinaryPrediction( y, 0.33 );`

`yH13 = distortBinaryPrediction( y, 0.34 );`

`yH14 = distortBinaryPrediction( y, 0.35 );`

`yH15 = distortBinaryPrediction( y, 0.36 );`

`yH16 = distortBinaryPrediction( y, 0.37 );`



#== split generated prediction set into disjoint sets eval and train==#

`limit         = round( Int64, 0.7 * size( y, 1 ) );` 

`predictions   = DataFrame( h1=yH1, h2=yH2, h3=yH3, h4=yH4, h5=yH5, h6=yH6, h7=yH7, h8=yH8,      h9=yH9, h10=yH10, h11=yH11, h12=yH12, h13=yH13, h14=yH14, h15=yH15, h16=yH16 );`

`predTraining    = predictions[1:limit,:];`

`predEval        = predictions[limit+1:end,:];`

`predMatTraining = convert( Matrix{Float64}, predTraining );`

`predMatEval     = convert( Matrix{Float64}, predEval );`

`errMatTraining  = ( repeat( Float64.( y[1:limit] ),outer = [1,size(predictions,2)] ) .- predMatTraining ).^2;`

`errMatTraining  = convert( Matrix{Float64}, errMatTraining );`

`sampleSize      = 32`

`nrRuns          = 100000`

`α_              = 1.0`



#== use bootstrap correlation algorithm to estimate the model posterior  distribution ==#

`P = bootstrapPosteriorCorEstimation( predictions, y, sampleSize, nrRuns );`



#== use bootstrap algorithm to estimate the model posterior distribution ==#

`p = bootstrapPosteriorEstimation( Matrix( errMatTraining ), sampleSize, nrRuns );` 



#== use Dirichletian algorithm to estimate the model posterior distribution ==#

`P = dirichletPosteriorEstimation( errMatTraining, nrRuns, α_ );`



#== use T-Distribution algorithm to estimate the model posterior distribution ==#

`P = TDistPosteriorEstimation( errMatTraining, nrRuns );`



#== make ensemble prediction ==#

`prediction = predictEnsemble( predictionsEval, p );`





**supported problems per algorithm**



|   algorithm    | univariate Classification | multivariate Classification | univariate Regression | multivariate Classification |
|:--------------:|:-------------------------:|:---------------------------:|:---------------------:|:---------------------------:|
| bootstrap      |            yes            |            yes              |         yes           |            yes              |
| bootstrap cor. |            yes            |            no               |         yes           |            no               |
| dirichletian   |    yes, only {0,1}-loss   |     yes, only {0,1}-loss    |         no            |            no               |
| t-distribution |            yes            |            yes              |         yes           |            yes              |

___



_**supported problems per fine tuning algorithms**_



|           algorithm           | univariate Classification | multivariate Classification | univariate Regression | multivariate Classification |
|:-----------------------------:|:-------------------------:|:---------------------------:|:---------------------:|:---------------------------:|
| δOptimizationMSE              |            yes            |             no              |           yes         |             no              |
| δOptimizationHinge            |            yes            |             no              |           no          |             no              |
| δOptimizationHingeRegularized |            yes            |             no              |           no          |             no              |
| δOptimizationMSERegularized   |            yes            |             no              |           yes         |             no              |