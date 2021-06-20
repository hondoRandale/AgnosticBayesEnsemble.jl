module AgnosticBayesEnsemble
  using Optim
  export TDistPosteriorEstimation, GMatrix, 
         dirichletPosteriorEstimation, 
         dirichletPosteriorEstimationV2, dirichletPosteriorEstimation!,
         metaParamSearchValidationDirichlet, 
         bootstrapPosteriorEstimation, bootstrapPosteriorEstimation!,
         bootstrapPosteriorCorEstimation,
         TDistPosteriorEstimation, TDistPosteriorEstimationReference,
         predictEnsemble,
         directOptimNaiveMSE, directOptimHinge,
         posteriorLinearBasis

  include( "argmaxProb.jl" )
  include( "argminProb.jl" )       
  include( "bootstrapPosteriorCorEstimation.jl" )
  include( "bootstrapPosteriorEstimation.jl" )
  include( "directSolution.jl" )
  include( "dirichletPosteriorEstimation.jl" )
  include( "predictEnsemble.jl" )
  
end






