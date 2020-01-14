include( "../src/bootstrapPosteriorEstimation.jl" )
using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running bootstrap algorithm unit tests." );   
predMatTraining, predMatEval, yTraining, yEval, errMatTraining = makeupPredictions();
posterior = bootstrapPosteriorEstimation( errMatTraining, 64, 10000 );
@test all( posterior .>= 0.0 );
@test all( posterior .<= 1.0 );
@test all( sum( posterior ) ≈ 1.0 );

posterior = zeros( Float64, size( predMatTraining, 2 ) );
bootstrapPosteriorEstimation!( errMatTraining, 100, 10000, posterior );
@test all( posterior .>= 0.0 );
@test all( posterior .<= 1.0 );
@test all( sum( posterior ) ≈ 1.0 );
