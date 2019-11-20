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
  
  """
  ## @param:  predictions    Matrix           - each column is prediction of one hypothesis
  ## @param:  weights        Array{Float64,1} - posterior p( h* = h | S )
  ## @brief:  perform bayesian ensemble prediction
  ## @return: Array{Float64,1} ensemble predicted values
  """
  function predictEnsemble( predictions::Matrix{Float64}, weights::Array{Float64,1} )
    res = zeros( Float64, size( predictions )[1] );
    for (i,col) in enumerate( eachcol( predictions ) )
      res .+= ( weights[i] .* col );
    end
    return res;
  end
end






