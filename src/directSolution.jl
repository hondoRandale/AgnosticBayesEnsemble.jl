include( "lossFunctions.jl" )
using Optim
using Statistics

  function objFunctionMSE( p::Vector{Float64}; predMat::Matrix{Float64}, t::Vector{Float64} )
    return mean( MSE( predMat * p, t ) );
  end

  function objFunctionHinge( p::Vector{Float64}; predMat::Matrix{Float64}, t::Vector{Float64} )
    return mean( hingeLoss( predMat * p, t ) );
  end

  function directOptimNaiveMSE( predMat::Matrix{Float64}, t::Vector{Float64}, p::Vector{Float64} )
    @assert size( t, 1 ) == size( t, 1 )  
    numberSamples = size( predMat, 1 );
    width         = size( predMat, 2 );
    f(p)          = objFunctionMSE( p,predMat=predMat, t=t )
    resultObject  = optimize( f, p );
    Optim.minimizer( resultObject )
  end

function directOptimHinge( predMat::Matrix{Float64}, t::Vector{Float64}, p::Vector{Float64} )
  @assert size( t, 1 ) == size( t, 1 ) 
  numberSamples = size( predMat, 1 );
  width         = size( predMat, 2 );
  f(p)          = objFunctionHinge( p,predMat=predMat, t=t );
  resultObject  = optimize( f, p );
  Optim.minimizer( resultObject )
end

function gMSE!( δ::Vector{Float64}, p::Vector{Float64}; predictions::Matrix{Float64}, y::Vector{Float64} )
  len           = size( predictions, 1 ); 
  width         = size( p, 1 );
  res           = zeros( Float64, width );
  predictionEns = predictions * p ; 
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
  g(p)            = gMSE!( δ, p, predictions=predMat, y=groundTruthMat );
  resultObject    = optimize( f, g!, lower, upper, p, Fminbox( inner_optimizer ) );
  Optim.minimizer( resultObject )
end


