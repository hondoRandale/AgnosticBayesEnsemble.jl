include( "AgnosticBayesEnsemble.jl" )
using DataFrames
using Random
using Statistics
using Optim

function distortBinaryPrediction( y::BitArray{1}, distortionFactor::Float64 )
  res          = deepcopy( y );   
  indices      = rand( 1:1:size( y, 1 ), round( Int64, distortionFactor * size( y, 1 ) ) );
  res[indices] = .!y[indices];
  return res;
end   

n    = 100000;
y    = Bool.( rand( 0:1,n ) );
yH1  = distortBinaryPrediction( y, 0.20 );
yH2  = distortBinaryPrediction( y, 0.20 );
yH3  = distortBinaryPrediction( y, 0.20 );
yH4  = distortBinaryPrediction( y, 0.25 );
yH5  = distortBinaryPrediction( y, 0.25 );
yH6  = distortBinaryPrediction( y, 0.25 );
yH7  = distortBinaryPrediction( y, 0.25 );
yH8  = distortBinaryPrediction( y, 0.25 );
yH9  = distortBinaryPrediction( y, 0.25 );
yH10 = distortBinaryPrediction( y, 0.25 );
yH11 = distortBinaryPrediction( y, 0.25 );
yH12 = distortBinaryPrediction( y, 0.30 );
yH13 = distortBinaryPrediction( y, 0.30 );
yH14 = distortBinaryPrediction( y, 0.30 );
yH15 = distortBinaryPrediction( y, 0.30 );
yH16 = distortBinaryPrediction( y, 0.30 );

predictions           = DataFrame( h1=yH1, h2=yH2, h3=yH3, h4=yH4, h5=yH5, h6=yH6, h7=yH7, h8=yH8, h9=yH9, h10=yH10, h11=yH11, h12=yH12, h13=yH13, h14=yH14, h15=yH15, h16=yH16 );
predMat               = Matrix( predictions );
errMat                = ( repeat(  Float64.( y ),outer = [1,size(predictions,2)] ) .- predictions ).^2
p                     = AgnosticBayesEnsemble.bootstrapPosteriorEstimation( Matrix( errMat ), 25, 10000 );
pD                    = AgnosticBayesEnsemble.dirichletPosteriorEstimation( Matrix( errMat ), nrRuns=10000, α_=0.1 );
pC                    = AgnosticBayesEnsemble.bootstrapPosteriorCorEstimation( convert( Matrix{Float64}, predMat ) , Float64.(y), 0.20, 10000 );
predictionEnsB        = round.( Int64, AgnosticBayesEnsemble.predictEnsemble( convert(Matrix{Float64},predMat), p ) );
predictionEnsD        = round.( Int64, AgnosticBayesEnsemble.predictEnsemble( convert(Matrix{Float64},predMat), pD ) );
yPredictionEnsBC      = round.( Int64, AgnosticBayesEnsemble.predictEnsemble( convert(Matrix{Float64},predMat), pC ) );

pU = zeros( Float64, 16 );
fill!( pU, 1/16 );
yUniform              = round.( AgnosticBayesEnsemble.predictEnsemble( convert( Matrix{Float64},predMat ), pU ) );

sum( y .== yH1 )
sum( y .== yH2 )
sum( y .== yH3 )
sum( y .== yH4 )
sum( Int64.(y) .== yPredictionEnsBC )
sum( Int64.(y) .== predictionEnsD ) 
sum( Int64.(y) .== predictionEnsB )
sum( Int64.(y) .== yUniform )

predictions = DataFrame( h1=yPredictionEnsBC, h2=predictionEnsD, h3=predictionEnsB );
predMat     = convert( Matrix{Float64}, predictions );
errMat      = ( repeat(  Float64.( y ),outer = [1,size(predictions,2)] ) .- predictions ).^2;

p           = AgnosticBayesEnsemble.bootstrapPosteriorEstimation( Matrix( errMat ), 0.25, 10000 );
pD          = AgnosticBayesEnsemble.dirichletPosteriorEstimation( Matrix( errMat ), nrRuns=10000, α_=0.1 );
pC          = AgnosticBayesEnsemble.bootstrapPosteriorCorEstimation( convert( Matrix{Float64}, predMat ) , Float64.(y), 0.20, 10000 );
## minimize err          enforce distribution axiom    
f( x ) = ( Ω * x[1:end-1] .- y ).^2 + λ * ( sum( x ) - 1.0 )^2       
x0 = zeros( Float64, d );
fill!( x0, 1/d );
optimize( f, x0 )









n    = 100000;
y    = Bool.( rand( 0:1,n ) );
yH1  = distortBinaryPrediction( y, 0.20 );
yH2  = distortBinaryPrediction( y, 0.21 );
yH3  = distortBinaryPrediction( y, 0.22 );
yH4  = distortBinaryPrediction( y, 0.23 );
yH5  = distortBinaryPrediction( y, 0.24 );
yH6  = distortBinaryPrediction( y, 0.24 );
yH7  = distortBinaryPrediction( y, 0.26 );
yH8  = distortBinaryPrediction( y, 0.27 );
yH9  = distortBinaryPrediction( y, 0.28 );
yH10 = distortBinaryPrediction( y, 0.29 );
yH11 = distortBinaryPrediction( y, 0.30 );
yH12 = distortBinaryPrediction( y, 0.33 );
yH13 = distortBinaryPrediction( y, 0.34 );
yH14 = distortBinaryPrediction( y, 0.35 );
yH15 = distortBinaryPrediction( y, 0.36 );
yH16 = distortBinaryPrediction( y, 0.37 );

limit           = round( Int64, 0.7 * size( y, 1 ) ); 
predictions     = DataFrame( h1=yH1, h2=yH2, h3=yH3, h4=yH4, h5=yH5, h6=yH6, h7=yH7, h8=yH8, h9=yH9, h10=yH10, h11=yH11, h12=yH12, h13=yH13, h14=yH14, h15=yH15, h16=yH16 );
predTraining    = predictions[1:limit,:];
predEval        = predictions[limit+1:end,:];
predMatTraining = convert( Matrix{Float64}, predTraining );
predMatEval     = convert( Matrix{Float64}, predEval );
errMatTraining  = ( repeat(  Float64.( y ),outer = [1,size(predictions,2)] ) .- predictions ).^2;



## learning curve bootstrap cor
precision = zeros( Float64, 1000 )
@showprogress 1 "Computing..." for (i,sampleSize) in enumerate( collect( 1:10:10000 ) )
  p              = bootstrapPosteriorCorEstimation( Matrix( predMatTraining ), Float64.(y[1:limit]), sampleSize, 50000 );
  predictionEnsB = round.( Int64, predictEnsemble( convert(Matrix{Float64},predMatEval), p ) );
  precision[i]   = sum( Int64.(y[limit+1:end]) .== predictionEnsB ) / size( predictionEnsB, 1 )
end

## learning curve bootstrap
precision = zeros( Float64, 150 )
for (i,sampleSize) in enumerate( collect( 1:1:150 ) )
  p              = bootstrapPosteriorEstimation( Matrix( errMatTraining ), sampleSize, 20000 );
  predictionEnsB = round.( Int64, predictEnsemble( convert(Matrix{Float64},predMatEval), p ) );
  precision[i]   = sum( Int64.(y[limit+1:end]) .== predictionEnsB ) / size( predictionEnsB, 1 )
end

## learning curve dirichlet
precision = zeros( Float64, size( collect( 0.005:1.0:40.0 ) ,1 ) );
for (i,α) in enumerate( collect( 0.005:1.0:40.0 ) )
  p              = dirichletPosteriorEstimation( Matrix( errMatTraining ), nrRuns=30000, α_=α );
  predictionEnsD = round.( Int64, predictEnsemble( convert(Matrix{Float64},predMatEval), p ) );
  precision[i]   = sum( Int64.(y[limit+1:end]) .== predictionEnsD ) / size( predictionEnsD, 1 );
end



pU = zeros( Float64, 16 );
fill!( pU, 1/16 );
yUniform = round.( predictEnsemble( convert( Matrix{Float64},predMatEval ), pU ) );
sum( Int64.(y[limit+1:end]) .== yUniform ) / size( y[limit+1:end], 1 )
