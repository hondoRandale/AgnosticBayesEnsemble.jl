include( "../src/gradientDescentOptimizePosterior.jl" )
using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running gradient descent algorithms unit tests." );
predMatTraining, predMatEval, tTraining, tEval, errMatTraining = makeupPredictions();
d         = size( predMatTraining, 2 );
YHopfield = deepcopy( predMatTraining );
toHopfieldEncoding!( YHopfield,predMatTraining );
tHopfield = deepcopy( tTraining );
toHopfieldEncoding!( tHopfield, tTraining );

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

resultDF, parameterEvalDf = δTuneHingeMeta(;posterior=posteriorStart, predMat=YHopfield, T=tHopfield, nrRunsRange=(3.0,10.0), αRange=(0.0,4.0), βRange=(0.0,4.0), relEntropyRange=(0.65,0.999), generations=2, siblings=100 );
resultDF, parameterEvalDf = δTuneMSEMeta(;posterior=posteriorStart, predMat=YHopfield, T=tHopfield, nrRunsRange=(3.0,10.0), αRange=(0.0,4.0), βRange=(0.0,4.0), relEntropyRange=(0.65,0.999), generations=2, siblings=100 );

#** Expirement3: train randomly initialized with MSE  **#
posterior          = randPosterior( d );
solutionRefinedMSE = δOptimizationMSE( posterior, YHopfield, tHopfield, 20 );
mseRand            = mean( MSE.( predMat * posterior, T ) );
mseMSE             = mean( MSE.( predMat * solutionRefinedMSE, T ) )
@test mseRand > mseMSE    
##show( resultDF )
##how( parameterEvalDf )



