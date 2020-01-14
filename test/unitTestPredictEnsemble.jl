using Test
include( "../src/predictEnsemble.jl" )
println( "running predictEnsemble unit tests." );

predictions = Vector{Matrix{Float64}}();
pred_1      = ones( 100, 3 );
pred_2      = 2.0 * ones( 100, 3 );
pred_3      = 3.0 * ones( 100, 3 );
push!( predictions, pred_1 );
push!( predictions, pred_2 );
push!( predictions, pred_3 );
weights     = zeros( 3 );
fill!( weights, 1/3 );
yEns  = predictEnsemble( predictions, weights );
@test all( yEns .≈ 2.0 )