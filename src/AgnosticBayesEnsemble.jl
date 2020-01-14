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
         posteriorLinearBasis, toHopfieldEncoding!,
         δOptimizationHinge, δOptimizationHingeRegularized,
         δOptimizationMSE, δOptimizationMSERegularized,
         δTuneMSEMeta, δTuneHingeMeta

  include( "argmaxProb.jl" )
  include( "argminProb.jl" )       
  include( "bootstrapPosteriorCorEstimation.jl" )
  include( "bootstrapPosteriorEstimation.jl" )
  include( "directSolution.jl" )
  include( "dirichletPosteriorEstimation.jl" )
  include( "gradientDescentOptimizePosterior.jl" )
  include( "predictEnsemble.jl" )
  include( "TDistPosteriorEstimation.jl" )
  
end






