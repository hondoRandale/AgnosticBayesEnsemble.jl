using ProgressMeter
using LinearAlgebra

  """
      bootstrapPosteriorCorEstimation( predictions::Matrix{Float64}, y::Array{Float64,1}, samplingBatchSize::Int64, nrRuns::Int64 )




      compute posterior p( h* = h | S ).
      #Arguments
      - `Y::Matrix{Float64}`: each column is the prediction error of one hypothesis.
      - `t::Vector{Float64}`: label vector.
      - `nrRuns::Int64`:      number of passes over predictions.
      - `minVal::Float64`:    minimum value of α.
      - `maxVal::Float64`:    maximum value of α.
      - `nSteps::Int64`:      number of steps between min and max val.
      - `holdout::Float64`:   percentage used in holdout.
      - `lossFunc`:           error function handle.
      #Return
      - `Float64`:            Best found meta parameter α. 
  """
  function bootstrapPosteriorCorEstimation( predictions::Matrix{Float64}, y::Array{Float64,1}, samplingBatchSize::Int64, nrRuns::Int64 )
    len      = size( predictions )[1];
    width    = size( predictions )[2];
    lenCache = samplingBatchSize;
    res      = zeros( Int64, width );
    for i in 1:1:nrRuns
      samplingIndexCache = rand( 1:size( predictions, 1 ), lenCache );
      label              = argmax( ( transpose( predictions[samplingIndexCache,:] ) * y[samplingIndexCache]  ) );
      res[label]        += 1; 
    end
    return  res ./ nrRuns; 
  end
  
  """
      bootstrapPosteriorCorEstimation( predictions::Matrix{Float64}, y::Matrix{Float64}, samplingFactor::Float64, nrRuns::Int64 )




      compute posterior p( h* = h | S )
  ## @param:  predictions    Matrix           - each column is prediction of one hypothesis
  ## @param:  samplingFactor Float64          - relative samples taken per iteration 
  ## @param:  nrRuns         Int64            - number of iterations over entire set
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
  """
  function bootstrapPosteriorCorEstimation( predictions::Matrix{Float64}, y::Matrix{Float64}, samplingFactor::Float64, nrRuns::Int64 )
    len      = size( predictions )[1];
    width    = size( predictions )[2];
    lenCache = round( Int64, samplingFactor * len );
    res      = zeros( Int64, width );
    for i in 1:1:nrRuns
      samplingIndexCache = rand( 1:size( predictions, 1 ), lenCache );
      θ                  = sum( y[samplingIndexCache,:] .* y[samplingIndexCache,:], 2 );
      label              = argmax( mean( transpose( predictions[samplingIndexCache,:] ) * y[samplingIndexCache,:] ./ θ , dims=2 ), dims=1 )[1,:] ;
      res[label]        += 1; 
    end
    return  res ./ nrRuns; 
  end