using Optim
using Statistics

  """
  @param:  p        -  Vector{Float64}  - hypothesis mixing factors
  @param:  predMat  -  Matrix{Float64}  - each column represents predictions of one model
  @param:  t        -  Vector{Float64}  - label vector
  @brief:  evaluate MeanSquaredError under given params
  @return: MSE      - Float64
  """
  function objFunctionMSE( p::Vector{Float64}; predMat::Matrix{Float64}, t::Vector{Float64} )
    return mean( lossFunctions.MSE( predMat * p, t ) );
  end

  """
  @param:  p         -  Vector{Float64}  - hypothesis mixing factors
  @param:  predMat   -  Matrix{Float64}  - each column represents predictions of one model
  @param:  t         -  Vector{Float64}  - label vector
  @brief:  evaluate Hinge Error under given params
  @return: hingeLoss - Float64
  """
  function objFunctionHinge( p::Vector{Float64}; predMat::Matrix{Float64}, t::Vector{Float64} )
    return mean( lossFunctions.hingeLoss( predMat * p, t ) );
  end

  """
  @param:  predMat   -  Matrix{Float64}  - each column represents predictions of one model
  @param:  t         -  Matrix{Float64}  - label vector
  @param:  p         -  Vector{Float64}  - hypothesis mixing factors
  @brief:  fine tune solution p using MSE error function
  @return: improved solution Vector{Float64} 
  """
  function directOptimNaiveMSE( predMat::Matrix{Float64}, t::Vector{Float64}, p::Vector{Float64} )
    @assert size( t, 1 ) == size( t, 1 )  
    numberSamples = size( predMat, 1 );
    width         = size( predMat, 2 );
    f(p)          = objFunctionMSE( p,predMat=predMat, t=t )
    resultObject  = optimize( f, p );
    Optim.minimizer( resultObject )
  end

  """
  @param:  predMat   -  Matrix{Float64}  - each column represents predictions of one model
  @param:  t         -  Matrix{Float64}  - label vector
  @param:  p         -  Vector{Float64}  - hypothesis mixing factors
  @brief:  fine tune solution p using hinge-Loss error function
  @return: improved solution Vector{Float64} 
  """
  function directOptimHinge( predMat::Matrix{Float64}, t::Vector{Float64}, p::Vector{Float64} )
    @assert size( t, 1 ) == size( t, 1 ) 
    numberSamples = size( predMat, 1 );
    width         = size( predMat, 2 );
    f(p)          = objFunctionHinge( p,predMat=predMat, t=t );
    resultObject  = optimize( f, p );
    Optim.minimizer( resultObject )
  end


