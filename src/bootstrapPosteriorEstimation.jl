using ProgressMeter

  """
      bootstrapPosteriorEstimation( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )




      compute posterior p( h* = h | S ).
      #Arguments
      - `errMat::Matrix{Float64}}`: each column is the prediction error of one hypothesis.
      - `samplingBatchSize::Int64`: sample size per main iteration.
      - `nrRuns::Int64`:            number of passes over predictions.
      #Return
      - `Vector{Float64}`:          Distribution p( h* = h | S ).
  """
  @views function bootstrapPosteriorEstimation( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )
    len                = size( errMat, 1 );
    width              = size( errMat, 2 );
    res                = zeros( Int, width );
    samplingIndexCache = zeros( Int, samplingBatchSize );
    @showprogress 1 "Computing..." for i in 1:1:nrRuns
      samplingIndexCache    .= rand( 1:size( errMat, 1 ), samplingBatchSize ) ;
      @inbounds label        = argminUniProb.argminProb( mean( errMat[samplingIndexCache,:], dims=1 )[1,:] );
      @inbounds res[label]  += 1; 
    end
    return  res ./ nrRuns; 
  end

  """
      bootstrapPosteriorEstimation!( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64, p::Array{Float64} )



      compute posterior p( h* = h | S ).
      #Arguments
      - `errMat::Matrix{Float64}}`: each column is the prediction error of one hypothesis.
      - `samplingBatchSize::Int64`: sample size per main iteration.
      - `nrRuns::Int64`:            number of passes over predictions.
      - `p::Vector{Float64}`:       resulting posterior p( h* = h | S ).
      #Return
      - `nothing`:                  nothing. 
  """
  function bootstrapPosteriorEstimation!( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64, p::Vector{Float64} )
    pB = bootstrapPosteriorEstimation( Matrix( errMat ), samplingBatchSize, nrRuns );
    for (i,val) in enumerate( pB )
      p[i] = val;
    end 
  end
