##include( "../src/AgnosticBayesEnsemble.jl" )
include( "../src/gradientDescentOptimizePosterior.jl" )
using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running gradient descent algorithms unit tests" );

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

lossLinearBasis = mean( lossFunctions.hingeLoss( yEnsLinearBasis, tHopfield ) );
lossHinge       = mean( lossFunctions.hingeLoss( yEnsHinge, tHopfield ) );
lossUni         = mean( lossFunctions.hingeLoss( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posteriorUni ), tHopfield ) );

@test lossUni > lossHinge

#** Expirement 2: start with linearBaseSolution normalized to distribution, hingeLoss **#
baseSolution              = posteriorLinearBasis( YHopfield, tHopfield );
γ                         = sum( baseSolution );
posteriorStart            = baseSolution ./ sum( baseSolution );
YHopfield                *= γ;
lossUni                   = mean( lossFunctions.hingeLoss( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posteriorUni ), tHopfield ) );
lossLinearBasisNormalized = mean( lossFunctions.hingeLoss( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posteriorStart ), tHopfield ) );
result = δOptimizationHinge( posteriorStart, YHopfield, tHopfield, 20 );

resultDF, parameterEvalDf = δTuneHingeMeta(;posterior=posteriorStart, predMat=YHopfield, T=tHopfield, nrRunsRange=(3.0,10.0), αRange=(0.0,4.0), βRange=(0.0,4.0), relEntropyRange=(0.65,0.999), generations=5, siblings=100 );
resultDF, parameterEvalDf = δTuneMSEMeta(;posterior=posteriorStart, predMat=YHopfield, T=tHopfield, nrRunsRange=(3.0,10.0), αRange=(0.0,4.0), βRange=(0.0,4.0), relEntropyRange=(0.65,0.999), generations=5, siblings=100 );

##show( resultDF )
##how( parameterEvalDf )



