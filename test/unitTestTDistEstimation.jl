include( "../src/TDistPosteriorEstimation.jl" )
using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running T-Distribution algorithm unit tests." );

predMatTraining, predMatEval, yTraining, yEval, errMatTraining = makeupPredictions();

posterior = TDistPosteriorEstimationReference( errMatTraining, 10000 );

@test all( posterior .>= 0.0 );
@test all( posterior .<= 1.0 );
@test all( sum( posterior ) ≈ 1.0 );

posterior = TDistPosteriorEstimation( errMatTraining, 1000000, κ_0 = 200000000.0, v_0 = 0.05, α = 0.00000000001 );

@test all( posterior .>= 0.0 );
@test all( posterior .<= 1.0 );
@test all( sum( posterior ) ≈ 1.0 );