include( "../src/AgnosticBayesEnsemble.jl" )
include( "../src/dirichletPosteriorEstimation.jl" )
using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running dirichletian algorithm unit tests" );
predMatTraining, predMatEval, tTraining, tEval, errMatTraining = makeupPredictions();
posterior = dirichletPosteriorEstimation( errMatTraining, 1000, 2.34 );

@test all( posterior .>= 0.0 )
@test all( posterior .<= 1.0 )
@test sum( posterior ) ≈ 1.0

posterior = dirichletPosteriorEstimationV2( errMatTraining, 100, 2.34, 10 );

@test all( posterior .>= 0.0 )
@test all( posterior .<= 1.0 )
@test sum( posterior ) ≈ 1.0

posterior = zeros( Float64, size( errMatTraining, 2 ) );
dirichletPosteriorEstimation!( errMatTraining, 1000, 2.34, posterior )

@test all( posterior .>= 0.0 )
@test all( posterior .<= 1.0 )
@test sum( posterior ) ≈ 1.0

αSequence, performance = metaParamSearchValidationDirichlet( predMatTraining, tTraining, 1000, 0.05, 10.0, 20, 0.25, lossFunctions.MSE );
best_index            = argmaxUniProb.argmaxProb( performance );

posterior_tuned = dirichletPosteriorEstimation( errMatTraining, 5000, αSequence[best_index] );
@test all( posterior_tuned .>= 0.0 )
@test all( posterior_tuned .<= 1.0 )
@test sum( posterior_tuned ) ≈ 1.0

posteriorUni = zeros( Float64, size( errMatTraining, 2 ) )
fill!( posteriorUni, 1 / size( errMatTraining, 2 ) );

yUni = round.( AgnosticBayesEnsemble.predictEnsemble( predMatEval, posteriorUni ) );
yDir = round.( AgnosticBayesEnsemble.predictEnsemble( predMatEval, posterior_tuned ) );

