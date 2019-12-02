include( "../src/argmaxProb.jl" )
using Test

  println( "running argmaxProb unit tests" );

  vec = repeat( [23.8], outer=[3 1] )[:,1];
  append!( vec, rand( 4.0:20.0, 100 ) );
  nrRuns = 100000;
  res    = zeros( Float64, 103 );
  for i in 1:1:nrRuns
    index = argmaxUniProb.argmaxProb( vec );
    res[index] += 1;
  end
  minVal = minimum( res[1:3] );
  maxVal = maximum( res[1:3] );

  @test maxVal / minVal > 0.95
  @test maxVal / minVal < 1.05 