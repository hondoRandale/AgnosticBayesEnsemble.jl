include( "../src/AgnosticBayesEnsemble.jl" )
include( "../src/dirichletPosteriorEstimation.jl" )
include( "../src/gradientDescentOptimizePosterior.jl" )
using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running dirichletian algorithm unit tests" );

function distortBinaryPrediction( y::BitArray{1}, distortionFactor::Float64 )
  res          = deepcopy( y );   
  indices      = rand( 1:1:size( y, 1 ), round( Int64, distortionFactor * size( y, 1 ) ) );
  res[indices] = .!y[indices];
  return res;
end   

n    = 100000;
t    = Bool.( rand( 0:1,n ) );
yH1  = distortBinaryPrediction( t, 0.20 );
yH2  = distortBinaryPrediction( t, 0.21 );
yH3  = distortBinaryPrediction( t, 0.22 );
yH4  = distortBinaryPrediction( t, 0.23 );
yH5  = distortBinaryPrediction( t, 0.24 );
yH6  = distortBinaryPrediction( t, 0.24 );
yH7  = distortBinaryPrediction( t, 0.26 );
yH8  = distortBinaryPrediction( t, 0.27 );
yH9  = distortBinaryPrediction( t, 0.28 );
yH10 = distortBinaryPrediction( t, 0.29 );
yH11 = distortBinaryPrediction( t, 0.30 );
yH12 = distortBinaryPrediction( t, 0.33 );
yH13 = distortBinaryPrediction( t, 0.34 );
yH14 = distortBinaryPrediction( t, 0.35 );
yH15 = distortBinaryPrediction( t, 0.36 );
yH16 = distortBinaryPrediction( t, 0.37 );
t    = Float64.( t );

predictions = DataFrame( h1=yH1, h2=yH2, h3=yH3, h4=yH4, h5=yH5, h6=yH6, h7=yH7, h8=yH8, h9=yH9, h10=yH10, h11=yH11, h12=yH12, h13=yH13, h14=yH14, h15=yH15, h16=yH16 );
d           = size( predictions, 2 );
Y           = convert( Matrix{Float64}, predictions );
errMat      = ( repeat(  Float64.( t ),outer = [1,size(Y,2)] ) .- Y ).^2;
posterior   = dirichletPosteriorEstimation( errMat, 10000, 2.34 );
@test sum( posterior ) ≈ 1.0
#== convert data to hopfield encoding ==#
YHopfield   = deepcopy( Y );
toHopfieldEncoding!( YHopfield, Y );
tHopfield   = deepcopy( t );
toHopfieldEncoding!( tHopfield, t );
sampleSize  = 32;
nrRuns      = 10000;

#== ensemble prediction bayesion posterior ==#
yEns      = sign.( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posterior ) ); 
mean( lossFunctions.hingeLoss( yEns, tHopfield ) );

#== ensemble prediction bayesion posterior fine tuned by gradient descend ==#
result     = δOptimizationHinge( posterior, YHopfield, tHopfield, 20 );
posterior2 = Optim.minimizer( result );
yEns2      = sign.( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posterior2 ) ); 
@test mean( lossFunctions.hingeLoss( yEns2, tHopfield ) ) < mean( lossFunctions.hingeLoss( yEns, tHopfield ) )

#== ensemble prediction bayesion posterior fine tuned by gradient descend regularized ==#
result     = δOptimizationHingeRegularized( posterior, YHopfield, tHopfield, 20, 2.5, 0.5, -0.8*log( 1/16 ) );
posterior3 = Optim.minimizer( result )
yEns3      = sign.( AgnosticBayesEnsemble.predictEnsemble( YHopfield, posterior3 ) ); 
##@test mean( hingeLoss( yEns3, tHopfield ) ) < mean( hingeLoss( yEns, tHopfield ) )