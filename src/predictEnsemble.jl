""" 
      predictEnsemble( predictions::Matrix{Float64}, weights::Vector{Float64} )
      



      perform bayesian ensemble prediction.
      #Arguments
      - `predictions::Matrix{Float64}`: each column is the prediction of one hypothesis.
      - `weights::Vector{Float64}`:     mixing weights.
      #Return
      - `Vector{Float64}`:              prediction y.
  """
  function predictEnsemble( predictions::Matrix{Float64}, weights::Vector{Float64} )
    return predictions * weights;
  end

  """ 
  predictEnsemble( predictions::Vector{Matrix{Float64}}, weights::Vector{Float64} )
  



  perform bayesian ensemble prediction.
  #Arguments
  - `predictions::Vector{Matrix{Float64}}`: each matrix is the prediction of one hypothesis.
  - `weights::Vector{Float64}`:             mixing weights.
  #Return
  - `Vector{Float64}`:                      prediction y.
  """
  function predictEnsemble( predictions::Vector{Matrix{Float64}}, weights::Vector{Float64} )
    len = size( weights, 1 );
    res = zeros( Float64, size( predictions[1], 1 ), size( predictions[1], 2 ) );
    for i in 1:1:len
      res .+= weights[i] .* predictions[i];
    end
    return res
  end