include( "AgnosticBayesEnsemble.jl" )
include( "gradientDescendOptimizePosterior.jl" )
using DataFrames
using Random
using Statistics
using StaticArrays
using Test
using Optim
using MultivariateStats


function distortBinaryPrediction( y::BitArray{1}, distortionFactor::Float64 )
  res          = deepcopy( y );   
  indices      = rand( 1:1:size( y, 1 ), round( Int64, distortionFactor * size( y, 1 ) ) );
  res[indices] = .!y[indices];
  return res;
end   

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
y    = Float64.( y )
limit           = round( Int64, 0.7 * size( y, 1 ) ); 
predictions     = DataFrame( h1=yH1, h2=yH2, h3=yH3, h4=yH4, h5=yH5, h6=yH6, h7=yH7, h8=yH8, h9=yH9, h10=yH10, h11=yH11, h12=yH12, h13=yH13, h14=yH14, h15=yH15, h16=yH16 );
predTraining    = predictions[1:limit,:];
predEval        = predictions[limit+1:end,:];
predMatTraining = convert( Matrix{Float64}, predTraining );
predMatEval     = convert( Matrix{Float64}, predEval );
errMatTraining  = ( repeat(  Float64.( y ),outer = [1,size(predictions,2)] ) .- predictions ).^2;
errMat          = errMatTraining;
errMat          = convert( Matrix{Float64}, errMat );
sampleSize      = 32;
nrRuns          = 200;

d         = size( predMatTraining, 2 );
YHopfield = deepcopy( predMatTraining );
toHopfieldEncoding!( YHopfield,predMatTraining );
tHopfield = deepcopy( y[1:limit] );
toHopfieldEncoding!( tHopfield, y[1:limit] );

#** Expirement 1: start with uniform distribution, hingeLoss **#
posteriorUni   = repeat( Array{Float64}([ 1/d ]), outer=[1, d] )[1,:];
posteriorHinge = repeat( Array{Float64}([ 1/d ]), outer=[1, d] )[1,:];
δOptimizationHinge( posteriorHinge, YHopfield, tHopfield, 20 );
yEnsLinearBasis = AgnosticBayesEnsemble.predictEnsemble( YHopfield, posteriorLinearBasis(YHopfield,tHopfield) );
indexPos        = ( yEnsLinearBasis .> 0.0 );
toHopfieldEncoding!( yEnsLinearBasis, Float64.( indexPos ) );
yEnsHinge = AgnosticBayesEnsemble.predictEnsemble( YHopfield, posteriorHinge );
indexPos  = ( yEnsHinge .> 0.0 );
toHopfieldEncoding!( yEnsHinge, Float64.( indexPos ) );

lossLinearBasis = mean( hingeLoss( yEnsLinearBasis, tHopfield ) );
lossHinge       = mean( hingeLoss( yEnsHinge, tHopfield ) );
lossUni         = mean( hingeLoss( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posteriorUni ), tHopfield ) );

@test lossUni > lossHinge

#** Expirement 2: start with linearBaseSolution normalized to distribution, hingeLoss **#
baseSolution              = posteriorLinearBasis( YHopfield, tHopfield );
γ                         = sum( baseSolution );
posteriorStart            = baseSolution ./ γ;
YHopfield                *= γ;
lossUni                   = mean( hingeLoss( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posteriorUni ), tHopfield ) );
lossLinearBasisNormalized = mean( hingeLoss( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posteriorStart ), tHopfield ) );
result = δOptimizationHinge( posteriorStart, YHopfield, tHopfield, 20 );

#== random init posterior n times ==#

using ProgressMeter
nrRuns = 200;
resMat = zeros( Float64, nrRuns, d );
@showprogress 1 "Computing..." for i in 1:1:nrRuns
  posterior   = randPosterior( d );
  result      = δOptimizationMSE( posterior, YHopfield, tHopfield, 20 );
  resMat[i,:] = Optim.minimizer( result );
end

μPosterior =  mean( resMat, dims=1 );
resMat   .-=  μPosterior;
σPosterior = std( resMat, dims=1 );
σPosterior ./ μPosterior;

@test sum( μPosterior ) ≈ 1.0

#== random init posterior n times ==#
nrRuns = 200;
resMat = zeros( Float64, nrRuns, width );

nrRunsRange     = ( 10, 30 );
αRange          = ( 0.0, 2.0 );
βRange          = ( 0.0, 2.0 );
relEntropyRange = ( 0.65, 0.99 );
generations     = 5;
siblings        = 400;
T               = tHopfield;


