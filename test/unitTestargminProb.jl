include( "../src/argminProb.jl" )
using Test

  println( "running argminProb unit tests" );

  vec = repeat( [2.0], outer=[3 1] )[:,1];
  append!( vec, rand( 4.0:20.0, 100 ) );
  nrRuns = 100000;
  res    = zeros( Float64, 103 );
  for i in 1:1:nrRuns
    index = argminUniProb.argminProb( vec );
    res[index] += 1;
  end
  minVal = minimum( res[1:3] );
  maxVal = maximum( res[1:3] );

  @test maxVal / minVal > 0.95
  @test maxVal / minVal < 1.05 

  len        = 20;
  lmatrix    = repeat( [2.0], outer=[len 3] );
  rmatrix    = rand( 4.0:20.0, len, 3 );
  vmatrix    = hcat( lmatrix, rmatrix );
  minIndices = argminProb( vmatrix );
  
  @test all( minIndices .>= 1 )
  @test all( minIndices .<= 3 )
