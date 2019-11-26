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
      directOptimNaiveMSE( predMat::Matrix{Float64}, t::Vector{Float64}, p::Vector{Float64} )




      compute refined solution _for_ mixing parameter p.
      #Arguments
      - `predMat::Matrix{Float64}`: each column is the prediction _of_ one hypothesis.
      - `t::Vector{Float64}`:       label vector.
      - `p::Vector{Float64}`:       initial solution for mixing coefficients.
      #Return
      - `Vector{Float64}`:          improved initial solution. 
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
      directOptimHinge( predMat::Matrix{Float64}, t::Vector{Float64}, p::Vector{Float64} )




      compute refined solution _for_ mixing parameter p.
      #Arguments
      - `predMat::Matrix{Float64}`: each column is the prediction _of_ one hypothesis.
      - `t::Vector{Float64}`:       label vector.
      - `p::Vector{Float64}`:       initial solution for mixing coefficients.
      #Return
      - `Vector{Float64}`:          improved initial solution. 
  """
  function directOptimHinge( predMat::Matrix{Float64}, t::Vector{Float64}, p::Vector{Float64} )
    @assert size( t, 1 ) == size( t, 1 ) 
    numberSamples = size( predMat, 1 );
    width         = size( predMat, 2 );
    f(p)          = objFunctionHinge( p,predMat=predMat, t=t );
    resultObject  = optimize( f, p );
    Optim.minimizer( resultObject )
  end


