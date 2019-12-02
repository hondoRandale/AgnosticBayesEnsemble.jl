using ProgressMeter
using LinearAlgebra

  """
      bootstrapPosteriorCorEstimation( predictions::Matrix{Float64}, t::Vector{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )




      compute posterior p( h* = h | S ).
      #Arguments
      - `predictions::Matrix{Float64}`: each column is the prediction of one hypothesis.
      - `t::Vector{Float64}`:           label vector.
      - `samplingBatchSize::Int64`:     sample size per main iteration.
      - `nrRuns::Int64`:                number of main  iterations.
      #Return
      - `Vector{Float64}`:              posterior p( h* = h | S ). 
  """
  function bootstrapPosteriorCorEstimation( predictions::Matrix{Float64}, t::Vector{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )
    len      = size( predictions )[1];
    width    = size( predictions )[2];
    lenCache = samplingBatchSize;
    res      = zeros( Int64, width );
    for i in 1:1:nrRuns
      samplingIndexCache = rand( 1:size( predictions, 1 ), lenCache );
      label              = argmaxUniProb.argmaxProb( ( transpose( predictions[samplingIndexCache,:] ) * t[samplingIndexCache]  ) );
      res[label]        += 1; 
    end
    return  res ./ nrRuns; 
  end
  

  """
      bootstrapPosteriorCorEstimation( predictions::Vector{Matrix{Float64}}, T::Matrix{Float64}, samplingFactor::Float64, nrRuns::Int64 )




      compute posterior p( h* = h | S ).
      #Arguments
      - `predictions::Matrix{Float64}`: each column is the prediction of one hypothesis.
      - `T::Matrix{Float64}`:           label matrix.
      - `samplingBatchSize::Int64`:     sample size per main iteration.
      - `nrRuns::Int64`:                number of main  iterations.
      #Return
      - `Vector{Float64}`:              posterior p( h* = h | S ). 
  """
  function bootstrapPosteriorCorEstimation( predictions::Vector{Matrix{Float64}}, T::Matrix{Float64}, samplingFactor::Float64, nrRuns::Int64 )
    len      = size( T, 1 );
    width    = size( predictions, 1 );
    lenCache = round( Int64, samplingFactor * len );
    res      = zeros( Int64, width );
    cache    = zeros( Float64, width );
    for i in 1:1:nrRuns
      samplingIndexCache = rand( 1:len, lenCache );
      for index in 1:1:width
        cache[index] = sum( transpose( predictions[i][samplingIndexCache,:] ) .* T[samplingIndexCache,:] )
      end
      label       = argmaxUniProb.argmaxProb( cache );
      res[label] += 1; 
    end
    return  res ./ nrRuns; 
  end