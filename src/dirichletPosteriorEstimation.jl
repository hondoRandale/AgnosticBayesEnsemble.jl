include( "argminProb.jl" )
include( "lossFunctions.jl" )

using Distributions
using LinearAlgebra
using ProgressMeter
using Match

function objFunctionASub( p::Vector{Float64}; predMat::Matrix{Float64}, y::Vector{Float64} )
  return 1.0 / ( ( sum( round.( Int64, predMat * p ) .== round.( Int64, y ) ) + 1e-8 ) / ( size( y, 1 ) ) );
end

function objectiveA( α::Float64, predMat::Matrix{Float64}, nrRuns::Int64, t::Vector{Float64} )
  errMat      = ( repeat(  Float64.( t ),outer = [1,size(predMat,2)] ) .- predMat ).^2;
  posterior   = dirichletPosteriorEstimation( errMat, nrRuns, α );
  yPrediction = predMat * posterior;
  return hingeLoss( yPrediction, t );
end

function gradient!( storage, x, predMat::Matrix{Float64}, t::Vector{Float64} )
  storage .= 2.0 * mean( (  predMat * x - t ) .* predMat, dims=1 )[1,:];
end

function metaParamSearchDirichletOptv2( initial_x::Vector{Float64}, predMat::Matrix{Float64}, nrRuns::Int64, t::Vector{Float64} )
  lowerBound = 1e-10;
  upperBound = 128.0;
  function f( α::Float64 )
    return objectiveA( α, predMat, nrRuns, t );
  end 
  function g!( storage, x )
    return gradient!( storage, x, predMat, t );
  end 
  od = OnceDifferentiable( f, g!, initial_x );
  return Optim.minimizer( optimize( od, initial_x, lowerBound, upperBound, Fminbox{GradientDescent}() ) );
end

  ## @param:  num     Int64 decimal number to be converted
  ## @brief:  convert Int64 to binary string
  ## @return: string - binary endoded number
  function tobin( num::Int64 )
    @match num begin
      0 => "0"
      1 => "1"
      _ => string( tobin( div( num, 2 ) ), mod( num, 2 ) )
    end
  end

  ## @param:  d     - Int64  number of hypothesis
  ## @brief:  compute transfoemation matrix G
  ## @return: transformation matrix G
  function GMatrix( d::Int64 )
    N   = 2^d;
    mat = Matrix{Float64}( undef, d, N );
    for row in 1:1:N
      mat[:,row] = parse.( Float64, split( lpad( tobin( row - 1 ), d, "0" ), "" ) );
    end
    return mat;
  end

  ## @param:  predictions    Matrix           - each column is prediction of one hypothesis
  ## @param:  G              Matrix           - transformation matrix computed by matrixG
  ## @param:  nrRuns         Int64            - number of passes over predictions
  ## @param:  α_             VariableRef      - scalar prior parameter
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
  function dirichletPosteriorEstimation( errMat::Matrix{Float64}, G::Matrix{Float64}, nrRuns::Int64, α_ )
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
      label       = argminProb( r );
      res[label] += 1.0;
    end
    return  res ./ nrRuns; 
  end

  ## @param:  predictions    Matrix           - each column is prediction of one hypothesis
  ## @param:  G              Matrix           - transformation matrix computed by matrixG
  ## @param:  nrRuns         Int64            - number of passes over predictions
  ## @param:  α_             Float64          - scalar prior parameter
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
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
      label       = argminProb( r );
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
      argminProb( r, labels );
      for pos in labels
        res[pos] += 1.0;
      end
    end
    return  res ./ sum( res ); 
  end
  
  ## @param:  errMat         Matrix           - each column is prediction of one hypothesis 
  ## @param:  nrRuns         Int64            - number of sampling runs
  ## @param:  α_             Float64          - scalar prior parameter
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
  function dirichletPosteriorEstimation( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64 )
    d = size( errMat )[2];
    G = GMatrix( d );
    return dirichletPosteriorEstimation( errMat, G, nrRuns, α_ );
  end

   ## @param: errMat         Matrix           - each column is prediction of one hypothesis 
  ## @param:  nrRuns         Int64            - number of sampling runs
  ## @param:  α_             Any              - scalar prior parameter
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
  function dirichletPosteriorEstimation( errMat::Matrix{Float64}, nrRuns::Int64, α_ )
    d = size( errMat )[2];
    G = GMatrix( d );
    return dirichletPosteriorEstimation( errMat, G, nrRuns, α_ );
  end

  ## @param:  errMat         Matrix           - each column is prediction of one hypothesis 
  ## @param:  nrRuns         Int64            - number of sampling runs
  ## @param:  α_             Float64          - scalar prior parameter
  ## @brief:  compute posterior p( h* = h | S )
  ## @return: posterior      Float{Float64,1} - Distribution p( h* = h | S )
  function dirichletPosteriorEstimationV2( errMat::Matrix{Float64}, nrRuns::Int64, α_::Float64, sampleSize::Int64 )
    d = size( errMat )[2];
    G = GMatrix( d );
    return dirichletPosteriorEstimationV2( errMat, G, nrRuns, α_, sampleSize );
  end

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