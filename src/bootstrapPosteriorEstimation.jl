using ProgressMeter
using StaticArrays

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
  function bootstrapPosteriorEstimation( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )
    len      = size( errMat, 1 );
    width    = size( errMat, 2 );
    res      = zeros( Int64, width );
    @showprogress 1 "Computing..." for i in 1:1:nrRuns
      samplingIndexCache = rand( 1:size( errMat, 1 ), samplingBatchSize ) ;
      label              = argminUniProb.argminProb( mean( errMat[samplingIndexCache,:], dims=1 )[1,:] );
      res[label]        += 1; 
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

  """
      bootstrapPosteriorEstimationP( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )



      compute posterior p( h* = h | S ).
      #Arguments
      - `errMat::Matrix{Float64}`:  each column is the prediction error of one hypothesis.
      - `samplingBatchSize::Int64`: label vector.
      - `nrRuns::Int64`:            number of passes over predictions.
      #Return
      - `Vector{Float64}`:          posterior p( h* = h | S ).
  """
  function bootstrapPosteriorEstimationP( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )
    tasks = Vector{Task}( undef, Threads.nthreads() );
    width = size( errMat, 2 );
    res   = Vector{ Vector{Float64} }( undef, Threads.nthreads() );
    p     = zeros( Float64, width );
    for i=1:1:Threads.nthreads()
      res[i]   = zeros( Float64, width );
      a()      = bootstrapPosteriorEstimation!( errMat, samplingBatchSize, nrRuns, res[i] );
      tasks[i] = Task( a );
    end
    for i in 1:1:Threads.nthreads()
      schedule( tasks[i] );
      yield();
    end
    for vec in res
      p .+= vec;
    end
    return p ./ Threads.nthreads();
  end