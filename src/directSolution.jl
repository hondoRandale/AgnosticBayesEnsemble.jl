include( "AgnosticBayesEnsemble.jl" )
using Optim
using Statistics

  function objFunction( p::Vector{Float64}; predMat::Matrix{Float64}, groundTruthMat::Vector{Float64} )
    return mean( ( predMat * p - groundTruthMat ) .^ 2 );
  end

  function objFunction( p::Vector{Float64}; predMat::Matrix{Float64}, groundTruthMat::Matrix{Float64} )
    return mean( ( predMat * p - groundTruthMat ) .^ 2 );
  end

  function directOptimNaive( predMat::Matrix{Float64}, groundTruthMat::Vector{Float64}, p::Vector{Float64} )
    @assert size( predMat, 1 ) == size( groundTruthMat, 1 )  
    numberSamples = size( predMat, 1 );
    width         = size( predMat, 2 );
    f(p)          = objFunction( p,predMat=predMat, groundTruthMat=groundTruthMat )
    resultObject  = optimize( f, p );
    Optim.minimizer( resultObject )
  end

function directOptim( predMat::Matrix{Float64}, groundTruthMat::Matrix{Float64} )
  @assert size( predMat, 1 ) == size( groundTruthMat, 1 )
  @assert size( predMat, 2 ) == size( groundTruthMat, 2 )  
  numberSamples = size( predMat, 1 );
  width         = size( predMat, 2 );
  f(p)          = objFunction( p,predMat=predMat, groundTruthMat=groundTruthMat );
  resultObject  = optimize( f, p );
  Optim.minimizer( resultObject )
end

function g!( δ::Vector{Float64}, p::Vector{Float64}; predictions::Matrix{Float64}, y::Vector{Float64} )
  len           = size( predictions, 1 ); 
  width         = size( p, 1 );
  res           = zeros( Float64, width );
  predictionEns = AgnosticBayesEnsemble.predictEnsemble( predictions, p ); 
  ##δ            .= 2 .* sum( ( predictionEns .- repeat(  Float64.( y ),outer = [1,width] ) ) .* predictions, dims=1 );
  δ            .= 2 * sum( ( predictionEns .- Float64.( y ) ) .* predictions, dims=1 )[1,:];                                                                                                                                                                                           
end


function directOptimDist1( predMat::Matrix{Float64}, groundTruthMat::Matrix{Float64} )
  d     = size( predMat, 2 );
  lower = zeros( Float64, d );
  upper = ones( Float64, d );
  p     = zeros( Float64, d );
  δ     = zeros( Float64, d );
  fill!( p, 1 / d );
  inner_optimizer = GradientDescent();
  f(p)            = objFunction( p,predMat=predMat, groundTruthMat=groundTruthMat )
  g(p)            = g!( δ, p, predictions=predMat, y=groundTruthMat );
  resultObject    = optimize( f, g!, lower, upper, p, Fminbox( inner_optimizer ) );
  Optim.minimizer( resultObject )
end

function directOptimDist1( predMat::Matrix{Float64}, groundTruthMat::Vector{Float64} )
  d     = size( predMat, 2 );
  lower = zeros( Float64, d );
  upper = ones( Float64, d );
  p     = zeros( Float64, d );
  fill!( p, 1 / d );
  inner_optimizer = GradientDescent();
  resultObject    = optimize( f, g!, lower, upper, p, Fminbox( inner_optimizer ) );
  Optim.minimizer( resultObject )
end

predMat        = convert( Matrix{Float64}, predictions );
groundTruthMat = Float64.( y );

pUni    = zeros( Float64, 16 );
fill!( pUni, 1 / 16 );

pDirect = directOptimNaive( predMat, groundTruthMat );
pBayes  = AgnosticBayesEnsemble.bootstrapPosteriorEstimation( errMat, 100, 10000 );
yBayes  = round.( Int64, AgnosticBayesEnsemble.predictEnsemble( predMat, pBayes ) );
yDirect = round.( Int64, AgnosticBayesEnsemble.predictEnsemble( predMat, pDirect ) );
yUni    = round.( Int64, AgnosticBayesEnsemble.predictEnsemble( predMat, pUni ) );


