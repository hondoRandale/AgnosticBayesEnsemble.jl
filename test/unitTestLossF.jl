include( "../src/lossFunctions.jl" )
using Statistics
using Test

println( "running loss functions' unit tests" );

#== testing for 0,1 labels error functions ==#
a = zeros( Float64, 10000 )
b = ones( Float64, 10000 )

@test mean( MSE( a, b ) ) == 1.0
@test mean( MSE( a, a ) ) == 0.0
@test mean( MSE( b, b ) ) == 0.0
@test mean( zeroOneLoss( a, b ) ) == 1.0
@test mean( zeroOneLoss( a, a ) ) == 0.0
@test mean( zeroOneLoss( b, b ) ) == 0.0 

#== testing for 1,-1 labels error functions ==#
a       = Bool.( rand( 0:1, 10000 ) );
b       = .!a;
index_a = ( a .== false );
a       = Float64.( a );
a[index_a] .= -1.0;
index_b     = ( b .== false );
b           = Float64.( b );
b[index_b] .= -1.0;
@test mean( hingeLoss( a, b ) ) == 2.0
@test mean( hingeLoss( a, a ) ) == 0.0
@test mean( hingeLoss( b, b ) ) == 0.0