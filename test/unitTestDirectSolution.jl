
include( "testingUtils.jl" )
include( "../src/bootstrapPosteriorEstimation.jl" )
include( "../src/directSolution.jl" )
using DataFrames

println( "running direct solution unit tests" );

predMatTraining, predMatEval, tTraining, tEval, errMatTraining = makeupPredictions()

predMat = convert( Matrix{Float64}, predMatTraining );
t       = Float64.( tTraining );

pUni    = zeros( Float64, 16 );
fill!( pUni, 1 / 16 );

pBayes       = bootstrapPosteriorEstimation( errMatTraining, 100, 10000 );
pDirectMSE   = directOptimNaiveMSE( predMat, t, pBayes );
pDirectHinge = directOptimHinge( predMat, t, pBayes );
yBayes       = predMatEval * pBayes;
yDirectMSE   = predMatEval * pDirectMSE;
yDirectHinge = predMatEval * pDirectHinge;
yUni         = predMatEval * pUni;

performanceBayes       = mean( lossFunctions.MSE( yBayes, tEval ) ); 
performanceDirectMSE   = mean( lossFunctions.MSE( yDirectMSE, tEval ) );
performanceDirectHinge = mean( lossFunctions.MSE( yDirectHinge, tEval ) );
performanceUni         = mean( lossFunctions.MSE( yUni, tEval ) );

#== test MSE optimization ==#
@test performanceBayes > performanceDirectMSE
@test performanceDirectHinge > performanceDirectMSE
@test performanceUni > performanceDirectMSE

performanceBayes       = mean( lossFunctions.hingeLoss( yBayes, tEval ) ); 
performanceDirectMSE   = mean( lossFunctions.hingeLoss( yDirectMSE, tEval ) );
performanceDirectHinge = mean( lossFunctions.hingeLoss( yDirectHinge, tEval ) ); 
performanceUni         = mean( lossFunctions.hingeLoss( yUni, tEval ) ); 

#== test hingeLoss optimization ==#
@test performanceBayes > performanceDirectHinge
@test performanceDirectMSE > performanceDirectHinge
@test performanceUni > performanceDirectHinge