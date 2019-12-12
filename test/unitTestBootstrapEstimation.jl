include( "../src/bootstrapPosteriorEstimation.jl" )
using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running bootstrap algorithm unit tests" );   
predMatTraining, predMatEval, yTraining, yEval, errMatTraining = makeupPredictions();
posterior = bootstrapPosteriorEstimation( errMatTraining, 64, 10000 );
@test all( posterior .>= 0.0 );
@test all( posterior .<= 1.0 );
@test all( sum( posterior ) ≈ 1.0 );