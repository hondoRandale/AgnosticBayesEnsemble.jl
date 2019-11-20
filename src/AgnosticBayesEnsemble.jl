module AgnosticBayesEnsemble
  using Optim
  export TDistPosteriorEstimation, GMatrix, 
         dirichletPosteriorEstimation, metaParamSearchDirichletOptv2, 
         dirichletPosteriorEstimationV2, dirichletPosteriorEstimation!,
         metaParamSearchValidationDirichlet, dirichletPosteriorEstimationPv1,
         bootstrapPosteriorEstimation, bootstrapPosteriorEstimation!,
         bootstrapPosteriorEstimationP,
         bootstrapPosteriorCorEstimation,
         TDistPosteriorEstimation, TDistPosteriorEstimationReference,
         predictEnsemble
  include( "argminProb.jl" )       
  include( "bootstrapPosteriorCorEstimation.jl" )
  include( "dirichletPosteriorEstimation.jl" )
  include( "gradientDescentOptimizePosterior.jl" )
  include( "bootstrapPosteriorEstimation.jl" )
  ##include( "lossFunctions.jl" )
  include( "TDistPosteriorEstimation.jl" )
  include( "directSolution.jl" )
  

  ## @param:  p       Vector{Float64}
  ## @param:  predMat Matrix{Float64}
  ## @param:  y       Vector{Float64}  
  ## @brief:  objective function measuring the performance of a solution
  ## @return: objective/error score 
  function objFunctionSquare( p::Vector{Float64}; predMat::Matrix{Float64}, t::Vector{Float64} )
    return mean( ( predMat * p - y ) .^ 2 );
  end

  ## @param:  p       Vector{Float64}
  ## @param:  predMat Matrix{Float64}
  ## @param:  y       Vector{Float64}  
  ## @brief:  objective function measuring the performance of a solution
  ## @return: objective/error score 
  function objFunctionPrec( p::Vector{Float64}; predMat::Matrix{Float64}, y::Vector{Float64} )
    return 1.0 / ( ( sum( round.( Int64, predMat * p ) .== round.( Int64, y ) ) + 1e-8 ) / ( size( y, 1 ) ) );
  end

  ## @param:  predMat             Matrix{Float64}
  ## @param:  groundTruthMat      Vector{Float64}
  ## @param:  p                   Vector{Float64}
  ## @brief:  refine results from agnostic bayes algorithms
  ## @return: Vector{Float64} refined solutions of p
  function refineNaiveSquare( predMat::Matrix{Float64}, groundTruthMat::Vector{Float64}, p::Vector{Float64} )
    @assert size( predMat, 1 ) == size( groundTruthMat, 1 )  
    numberSamples = size( predMat, 1 );
    width         = size( predMat, 2 );
    f(p)          = objFunctionSquare( p,predMat=predMat, t=t );
    resultObject  = optimize( f, p );
    Optim.minimizer( resultObject )
  end

  ## @param:  predMat             Matrix{Float64}
  ## @param:  groundTruthMat      Vector{Float64}
  ## @param:  p                   Vector{Float64}
  ## @brief:  refine results from agnostic bayes algorithms
  ## @return: Vector{Float64} refined solutions of p
  function refineNaivePrec( predMat::Matrix{Float64}, y::Vector{Float64}, p::Vector{Float64} )
    @assert size( predMat, 1 ) == size( y, 1 )  
    numberSamples = size( predMat, 1 );
    width         = size( predMat, 2 );
    f(p)          = objFunctionPrec( p,predMat=predMat, y=y );
    resultObject  = optimize( f, p );
    Optim.minimizer( resultObject );
  end


  ## @param:  predictions    Matrix           - each column is prediction of one hypothesis
  ## @param:  weights        Array{Float64,1} - posterior p( h* = h | S )
  ## @brief:  perform bayesian ensemble prediction
  ## @return: Array{Float64,1} ensemble predicted values
  function predictEnsemble( predictions::Matrix{Float64}, weights::Array{Float64,1} )
    res = zeros( Float64, size( predictions )[1] );
    for (i,col) in enumerate( eachcol( predictions ) )
      res .+= ( weights[i] .* col );
    end
    return res;
  end
end






