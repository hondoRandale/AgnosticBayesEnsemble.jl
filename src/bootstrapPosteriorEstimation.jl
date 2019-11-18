include( "argminProb.jl" )
using ProgressMeter
using StaticArrays

  """
  ## @param:  errMat         Matrix           - each column is prediction of one hypothesis
  ## @param:  samplingFactor Float64          - relative samples taken per iteration 
  ## @param:  nrRuns         Int64            - number of iterations over entire set
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
  """
  function bootstrapPosteriorEstimation( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )
    len      = size( errMat, 1 );
    width    = size( errMat, 2 );
    res      = zeros( Int64, width );
    @showprogress 1 "Computing..." for i in 1:1:nrRuns
      samplingIndexCache = rand( 1:size( errMat, 1 ), samplingBatchSize ) ;
      label              = argminProb( mean( errMat[samplingIndexCache,:], dims=1 )[1,:] );
      res[label]        += 1; 
    end
    return  res ./ nrRuns; 
  end

  """
  ## @param:  errMat         Matrix           - each column is prediction of one hypothesis
  ## @param:  samplingFactor Float64          - relative samples taken per iteration 
  ## @param:  nrRuns         Int64            - number of iterations over entire set
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
  """
  function bootstrapPosteriorEstimation!( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64, p::Array{Float64} )
    pB = bootstrapPosteriorEstimation( Matrix( errMat ), samplingBatchSize, nrRuns );
    for (i,val) in enumerate( pB )
      p[i] = val;
    end 
  end

  """
  ## @param:  errMat         Matrix           - each column is prediction of one hypothesis
  ## @param:  samplingFactor Float64          - relative samples taken per iteration 
  ## @param:  nrRuns         Int64            - number of iterations over entire set
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
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