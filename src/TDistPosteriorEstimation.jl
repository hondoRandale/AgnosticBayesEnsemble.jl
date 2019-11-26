  using Distributions
  using ProgressMeter

  function metaParamSearchValidationTDist()
    
  end
  
  function metaParamSearchValidationTDistOptv1()
    
  end

  """
      TDistPosteriorEstimationReference( errMat::Matrix{Float64}, nrRuns::Int64 )




      compute posterior p( h* = h | S ).
      #Arguments
      - `errMat::Matrix{Float64}`: each column is the prediction error of one hypothesis.
      - `nrRuns::Int64`:                number of main  iterations.
      #Return
      - `Vector{Float64}`:              posterior p( h* = h | S ). 
  """
  function TDistPosteriorEstimationReference( errMat::Matrix{Float64}, nrRuns::Int64 )
    m   = size( errMat )[1];
    d   = size( errMat )[2];
    l_  = mean( errMat, dims=1 )[1,:];
    S   = cov( errMat, corrected=true );
    κ_0 = 1.0;
    v_0 = d;
    r_0 = 0.5 * ones( Float64, d );
    T_0 = 0.25 .* Matrix{Float64}( I, d, d );
    κ_m = κ_0 + m;
    v_m = v_0 + m;
    v_  = v_m - d + 1;
    r_m = ( κ_0 .* r_0 .+ m .* l_ ) ./ κ_m;
    T_m = Symmetric( T_0 .+ m .* S + m * ( κ_0 / κ_m ) .* ( r_0 - l_ ) * transpose( r_0 - l_ ) );
    @assert issymmetric( T_m ) 
    gaussianDist   = MvNormal( zeros( Float64, d ), T_m ./ ( κ_m * v_ ) );
    chiSquaredDist = Chisq( v_ ); 
    res            = zeros( Int64, d );
    @showprogress 1 "Computing..." for i in 1:1:nrRuns
      ## sample z from multivar gaussian
      z           = rand( gaussianDist );
      ## sample ϵ from univariate chi-squared
      ϵ           = rand( chiSquaredDist );
      r           =  r_m .+ z * sqrt( v_ / ϵ ); 
      label       = argminUniProb.argminProb( r );
      res[label] += 1;
    end
    return  res ./ nrRuns;
  end

  """
      TDistPosteriorEstimation( errMat::Matrix{Float64}, nrRuns::Int64; [κ_0::Float64=1.0] [, v_0::Float64=Float64( size( errMat, 2 ) )] [, α::Float64=0.5] [, β::Float64=0.25] )




      compute posterior p( h* = h | S ).
      #Arguments
      - `errMat::Matrix{Float64}`:                   each column is the prediction error of one hypothesis.
      - `nrRuns::Int64`:                             number of main  iterations.
      - `κ_0::Float64=1.0`:                          regularization param.
      - `v_0::Float64=Float64( size( errMat, 2 ) )`: regularization param.
      - `α::Float64=0.5`:                            regularization param.
      - `β::Float64=0.25`:                           regularization param.
      #Return
      - `Vector{Float64}`:                           posterior p( h* = h | S ). 
  """
  function TDistPosteriorEstimation( errMat::Matrix{Float64}, nrRuns::Int64; κ_0::Float64=1.0, v_0::Float64=Float64( size( errMat, 2 ) ), α::Float64=0.5, β::Float64=0.25 )
    m   = size( errMat )[1];
    d   = size( errMat )[2];
    l_  = mean( errMat, dims=1 )[1,:];
    S   = cov( errMat, corrected=true );
  
    r_0 = α * ones( Float64, d );
    T_0 = β .* Matrix{Float64}( I, d, d );
    κ_m = κ_0 + m;
    v_m = v_0 + m;
    v_  = v_m - d + 1;
    r_m = ( κ_0 .* r_0 .+ m .* l_ ) ./ κ_m;
    T_m = Symmetric( T_0 .+ m .* S + m * ( κ_0 / κ_m ) .* ( r_0 - l_ ) * transpose( r_0 - l_ ) );
    @assert issymmetric( T_m ) 
    gaussianDist   = MvNormal( zeros( Float64, d ), T_m ./ ( κ_m * v_ ) );
    chiSquaredDist = Chisq( v_ ); 
    res            = zeros( Int64, d );
    @showprogress 1 "Computing..." for i in 1:1:nrRuns
      ## sample z from multivar gaussian
      z           = rand( gaussianDist );
      ## sample ϵ from univariate chi-squared
      ϵ           = rand( chiSquaredDist );
      r           =  r_m .+ z * sqrt( v_ / ϵ ); 
      label       = argminProb( r );
      res[label] += 1;
    end
    return  res ./ nrRuns;
  end

  ## TDistPosteriorEstimation( errMat, 10000000, κ_0 = 2000000000.0, v_0 = 0.05, α = 0.000000000001 )