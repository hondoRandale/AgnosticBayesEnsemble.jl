include( "../src/bootstrapPosteriorCorEstimation.jl" )

using DataFrames
using Distributions
using Random
using Statistics
using Test
using Optim
using MultivariateStats

function distortVector( vec::Vector{Float64}, pct::Float64, σ::Float64 )
  res          = deepcopy( vec );  
  len          = size( vec, 1 );  
  indices      = rand( 1:1:len, round( Int64, pct*len ) );
  gaussUniDist = Normal( 0.0, σ )
  for i in indices
    res[i] = vec[i] + rand( gaussUniDist );
  end
  return res;
end

println( "running bootstrap correlation algorithm unit tests." );
predMatTraining, predMatEval, yTraining, yEval, errMatTraining = makeupPredictions();
posterior = bootstrapPosteriorCorEstimation( predMatTraining, yTraining, 64, 10000 );
@test all( posterior .>= 0.0 );
@test all( posterior .<= 1.0 );
@test all( sum( posterior ) ≈ 1.0 );

nSamples    = 10000;
X           = rand( Normal( 0.0, 1.5 ), nSamples );
t_1         = Float64[ ( 0.3 * x^3 + 4.0 ) for x in X ];
t_2         = Float64[ ( 0.2 * x^2 + 0.1 * x ) for x in X ];
t_3         = Float64[ ( 0.5 * x^3 + 3.4 * x^2 + 1.5 * x + 0.95 ) for x in X ];
TMat        = hcat( t_1, t_2, t_3 );
predictions = Vector{ Matrix{Float64} }();
mat1 = hcat( distortVector( t_1, 0.05, 1.85 ),
             distortVector( t_2, 0.30, 0.75 ),
             distortVector( t_3, 0.25, 0.5 ) );
push!( predictions, mat1 );

mat2 = hcat( distortVector( t_1, 0.35, 1.25 ),
             distortVector( t_2, 0.10, 0.95 ),
             distortVector( t_3, 0.45, 0.35 ) );
push!( predictions, mat2 );      

mat3 = hcat( distortVector( t_1, 1.35, 2.25 ),
             distortVector( t_2, 0.03, 0.07 ),
             distortVector( t_3, 0.25, 5.0 ) );
push!( predictions, mat3  );
nrRuns         = 5000;
samplingFactor = 0.4;
posterior      = bootstrapPosteriorCorEstimation( predictions, TMat, samplingFactor, nrRuns );
@test all( posterior .>= 0.0 );
@test all( posterior .<= 1.0 );
@test all( sum( posterior ) ≈ 1.0 );
