include( "lossFunctions.jl" )

using Distributions
using LinearAlgebra
using ProgressMeter
using Match

  """
      tobin( num::Int64 )




  """
  function tobin( num::Int64 )
    @match num begin
      0 => "0"
      1 => "1"
      _ => string( tobin( div( num, 2 ) ), mod( num, 2 ) )
    end
  end
  
  """
      GMatrix( d::Int64 )




      compute transformation matrix G.
      #Arguments
      - `d::Int64`:        number of hypothesis used for prediction.
      #Return
      - `Matrix{Float64}`: transformation matrix G. 
  """
  function GMatrix( d::Int64 )
    N   = 2^d;
    mat = Matrix{Float64}( undef, d, N );
    for row in 1:1:N
      mat[:,row] = parse.( Float64, split( lpad( tobin( row - 1 ), d, "0" ), "" ) );
    end
    return mat;
  end

  """
      dirichletPosteriorEstimation( errMat::Matrix{Float64}, G::Matrix{Float64}, nrRuns::Int64, α_ )




      compute posterior p( h* = h | S ).
      # Arguments
      - `errMat::Matrix{Float64}`: each column is the prediction of one hypothesis.
      - `nrRuns::Int64`:           number of sampling runs.
      - `α_::Float64`:             scalar prior parameter.
      - `sampleSize::Int64`:       number of samples per run.
      # Return
      - `Vector{Float64}`:         posterior distribution
  """
  function dirichletPosteriorEstimation( errMat::Matrix{Float64}, G::Matrix{Float64}, nrRuns::Int64, α_::Float64 )
    ## number of prediction models to combine
    m = size( errMat )[1];
    d = size( errMat )[2];
    ## number outcomes
    N = 2^d;
    ## counts outcomes 
    K   = zeros( Float64, N );
    ## init resulting posterior distribution
    res = zeros( Float64, d );
    ## multivariate prior parameter
    α = α_ * ones( Float64, N );
  
    ## compute count OF OCCURENCES K
    @showprogress 1 "Computing..." for i in 1:1:m
      col     = parse( Int64, join( string.( convert( Array{Int64,1}, errMat[i,:] ) ) ), base=2 ) + 1;
      K[col] += 1; 
    end
    K ./= sum( K ); 
  
    ## ML- Distribution
    βDist    = Beta( α_ * N, m );
    ## prior Distribution
    αDirDist = Dirichlet( α );
    ## ML- Distribution
    KDirDist = Dirichlet( K .+ 1e-9 );
  
    @showprogress 1 "Computing..." for i in 1:1:nrRuns
      θSample     = rand( βDist );
      αSample     = rand( αDirDist );
      KSample     = rand( KDirDist );
      q           = ( θSample .* αSample ) .+ ( ( 1 - θSample ) * KSample );
      r           = G * q;
      label       = argminUniProb.argminProb( r );
      res[label] += 1.0;
    end
    return  res ./ nrRuns; 
  end

  function dirichletPosteriorEstimationV2( errMat::Matrix{Float64}, G::Matrix{Float64}, nrRuns::Int64, α_::Float64, sampleSize::Int64 )
    ## number of prediction models to combine
    m = size( errMat, 1 );
    d = size( errMat, 2 );
    ## number outcomes
    N = 2^d;
    ## counts outcomes 
    K   = zeros( Float64, N );
    ## init resulting posterior distribution
    res = zeros( Float64, d );
    ## multivariate prior parameter
    α = α_ * ones( Float64, N );
  
    ## compute count OF OCCURENCES K
    for i in 1:1:m
      col     = parse( Int64, join( string.( convert( Array{Int64,1}, errMat[i,:] ) ) ), base=2 ) + 1;
      K[col] += 1; 
    end
    K ./= sum( K ); 
  
    ## ML- Distribution
    βDist    = Beta( α_ * N, m );
    ## prior Distribution
    αDirDist = Dirichlet( α );
    ## ML- Distribution
    KDirDist = Dirichlet( K .+ 1e-9 );
  
    labels = zeros( Int64, sampleSize );
    for i in 1:1:nrRuns
      θSample     = rand( βDist,    sampleSize );
      αSample     = rand( αDirDist, sampleSize );
      KSample     = rand( KDirDist, sampleSize );
      for i in 1:1:sampleSize
        for j in 1:1:N
          αSample[j,i] *= θSample[i];
          KSample[j,i] *= ( 1.0 - θSample[i] );
        end
      end
      q           = αSample .+ KSample;
      r           = G * q;
      argminUniProb.argminProb( r, labels );
      for pos in labels
        res[pos] += 1.0;
      end
    end
    return  res ./ sum( res ); 
  end

  """
      dirichletPosteriorEstimation( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64 )
      



      compute posterior p( h* = h | S ).
      # Arguments
      - `errMat::Matrix{Float64}`: each column is the prediction of one hypothesis.
      - `nrRuns::Int64`:           number of sampling runs.
      - `α_::Float64`:             scalar prior parameter.
      - `sampleSize::Int64`:       number of samples per run.
      # Return
      - `Vector{Float64}`:         posterior distribution

      
  """
  function dirichletPosteriorEstimation( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64 )
    d = size( errMat )[2];
    G = GMatrix( d );
    return dirichletPosteriorEstimation( errMat, G, nrRuns, α_ );
  end

  """
      dirichletPosteriorEstimationV2( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64, sampleSize::Int64 )
      compute posterior p( h* = h | S ).
      # Arguments
      - `errMat::Matrix{Float64}`: each column is the prediction of one hypothesis.
      - `nrRuns::Int64`:           number of sampling runs.
      - `α_::Float64`:             scalar prior parameter.
      - `sampleSize::Int64`:       number of samples per run.
      # Return
      - `Vector{Float64}`:         posterior distribution
  """
  function dirichletPosteriorEstimationV2( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64, sampleSize::Int64 )
    d = size( errMat )[2];
    G = GMatrix( d );
    return dirichletPosteriorEstimationV2( errMat, G, nrRuns, α_, sampleSize );
  end

  """
      dirichletPosteriorEstimation!( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64, p::Vector{Float64} )




      compute posterior p( h* = h | S ).
      #Arguments
      - `errMat::Matrix{Float64}`: each column is the prediction error of one hypothesis.
      - `nrRuns::Int64`:           number of passes over predictions.
      - `α_::Float64`:             meta parameter value.
      - `p::Vector{Float64}`:      return value posterior p( h* = h | S ).
      #Return
      - `Float64`:            Best found meta parameter α. 
  """
  function dirichletPosteriorEstimation!( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64, p::Vector{Float64} )
    pRes = dirichletPosteriorEstimation( errMat, nrRuns, α_ );
    for (i,val) in enumerate( pRes )
      p[i] = val;
    end
  end

  function dirichletPosteriorEstimationPv1( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64 )
    tasks = Vector{Task}( undef, Threads.nthreads() );
    width = size( errMat, 2 );
    res   = Vector{ Vector{Float64} }( undef, Threads.nthreads() );
    p     = zeros( Float64, width );
    for i=1:1:Threads.nthreads()
      res[i]   = zeros( Float64, width );
      a()      = dirichletPosteriorEstimation!( errMat, nrRuns, α_, res[i] );
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
  
  """
      metaParamSearchValidationDirichlet( Y::Matrix{Float64}, t::Vector{Float64}, nrRuns::Int64, minVal::Float64, maxVal::Float64, nSteps::Int64, holdout::Float64, lossFunc )    




      compute best α parameter regarding predictive performance.
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
  function metaParamSearchValidationDirichlet( Y::Matrix{Float64}, t::Vector{Float64}, nrRuns::Int64, minVal::Float64, maxVal::Float64, nSteps::Int64, holdout::Float64, lossFunc )
    @assert nrRuns > 0
    @assert maxVal > minVal
    @assert minVal >= 0.0 && maxVal > 0.0
    @assert nSteps > 1
    @assert size( Y, 1 ) == size( t, 1 )
    @assert holdout * size( Y, 1 ) > 1.0 
  
    limit      = Int64( holdout * size( Y, 1 ) );
    YTrain     = Y[limit+1:end,:];
    tTrain     = t[limit+1:end];
    YEval      = Y[1:limit,:];
    tEval      = t[1:limit];
    errMat     = ( repeat(  Float64.( tTrain ), outer = [ 1, size( YTrain, 2 ) ] ) .- YTrain ).^2;
    αSequence  = collect( minVal:(maxVal-minVal)/nSteps:maxVal );
    performace = Float64[]   
    for α in αSequence
      posterior   = dirichletPosteriorEstimation( errMat, nrRuns, α );
      yPrediction = YEval * posterior;
      push!( performace, mean( lossFunc( yPrediction, tEval ) ) );
    end  
    return αSequence, performace
  end