include( "../src/AgnosticBayesEnsemble.jl" )
include( "../src/dirichletPosteriorEstimation.jl" )
include( "../src/gradientDescentOptimizePosterior.jl" )
using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running dirichletian algorithm unit tests" );
predMatTraining, predMatEval, yTraining, yEval, errMatTraining = makeupPredictions();
posterior   = dirichletPosteriorEstimation( errMatTraining, 10000, 2.34 );
@test sum( posterior ) ≈ 1.0

#== convert data to hopfield encoding ==#
YHopfield   = deepcopy( predMatEval );
toHopfieldEncoding!( YHopfield, predMatEval );
tHopfield   = deepcopy( yEval );
toHopfieldEncoding!( tHopfield, yEval );

#== ensemble prediction bayesion posterior ==#
yEns      = sign.( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posterior ) ); 
mean( lossFunctions.hingeLoss( yEns, tHopfield ) );

#== ensemble prediction bayesion posterior fine tuned by gradient descend ==#
result     = δOptimizationHinge( posterior, YHopfield, tHopfield, 20 );
posterior2 = Optim.minimizer( result );
yEns2      = sign.( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posterior2 ) ); 
@test mean( lossFunctions.hingeLoss( yEns2, tHopfield ) ) < mean( lossFunctions.hingeLoss( yEns, tHopfield ) )

#== ensemble prediction bayesion posterior fine tuned by gradient descend regularized ==#
result     = δOptimizationHingeRegularized( posterior, YHopfield, tHopfield, 20, 2.5, 0.5, -0.8*log( 1/16 ) );
posterior3 = Optim.minimizer( result )
yEns3      = sign.( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posterior3 ) ); 
##@test mean( hingeLoss( yEns3, tHopfield ) ) < mean( hingeLoss( yEns, tHopfield ) )