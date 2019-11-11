include( "AgnosticBayesEnsemble.jl" )
using DataFrames
using Random
using Statistics
using StaticArrays
using Optim
using MultivariateStats


function distortBinaryPrediction( y::BitArray{1}, distortionFactor::Float64 )
  res          = deepcopy( y );   
  indices      = rand( 1:1:size( y, 1 ), round( Int64, distortionFactor * size( y, 1 ) ) );
  res[indices] = .!y[indices];
  return res;
end   

n    = 100000;
y    = Bool.( rand( 0:1,n ) );
yH1  = distortBinaryPrediction( y, 0.20 );
yH2  = distortBinaryPrediction( y, 0.21 );
yH3  = distortBinaryPrediction( y, 0.22 );
yH4  = distortBinaryPrediction( y, 0.23 );
yH5  = distortBinaryPrediction( y, 0.24 );
yH6  = distortBinaryPrediction( y, 0.24 );
yH7  = distortBinaryPrediction( y, 0.26 );
yH8  = distortBinaryPrediction( y, 0.27 );
yH9  = distortBinaryPrediction( y, 0.28 );
yH10 = distortBinaryPrediction( y, 0.29 );
yH11 = distortBinaryPrediction( y, 0.30 );
yH12 = distortBinaryPrediction( y, 0.33 );
yH13 = distortBinaryPrediction( y, 0.34 );
yH14 = distortBinaryPrediction( y, 0.35 );
yH15 = distortBinaryPrediction( y, 0.36 );
yH16 = distortBinaryPrediction( y, 0.37 );

limit           = round( Int64, 0.7 * size( y, 1 ) ); 
predictions     = DataFrame( h1=yH1, h2=yH2, h3=yH3, h4=yH4, h5=yH5, h6=yH6, h7=yH7, h8=yH8, h9=yH9, h10=yH10, h11=yH11, h12=yH12, h13=yH13, h14=yH14, h15=yH15, h16=yH16 );
predTraining    = predictions[1:limit,:];
predEval        = predictions[limit+1:end,:];
predMatTraining = convert( Matrix{Float64}, predTraining );
predMatEval     = convert( Matrix{Float64}, predEval );
errMatTraining  = ( repeat(  Float64.( y ),outer = [1,size(predictions,2)] ) .- predictions ).^2;
errMat          = errMatTraining;
errMat          = convert( Matrix{Float64}, errMat );
sampleSize      = 32
nrRuns          = 100000



function do_work( e::Matrix{Float64}, sSize::Int64, nRuns::Int64, result::Channel, job::Channel )
  for job_id in job
    p = AgnosticBayesEnsemble.bootstrapPosteriorEstimation( Matrix( e ), sSize, nRuns );                
    put!( result, SVector{16,Float64}( p ) );
  end
end;

function do_work()
  for job_id in jobs
    p = bootstrapPosteriorEstimation( Matrix( errMat ), sampleSize, nrRuns );                
    put!( results, SVector{16,Float64}( p ) );
  end
end;

function make_jobs(n)
    for i in 1:n
        put!(jobs, i)
    end
end;

function make_jobs(n,c::Channel)
  for i in 1:n
    put!( c, i )
  end
end;
 
function bootstrapPosteriorEstimationP( errMat::Matrix{Float64}, sampleSize::Int64, nrRuns::Int64 )
  width   = size( predictions, 2 );
  jobs    = Channel{Int}( Threads.nthreads() );
  results = Channel{ SArray{Tuple{width},Float64,1,width} }( Threads.nthreads() );
  res     = Matrix{Float64}( undef, Threads.nthreads(), width );
  ##@async make_jobs( Threads.nthreads() );
  @async make_jobs( Threads.nthreads(), jobs );
  for i in 1:Threads.nthreads() 
    @async do_work( errMat, sampleSize, nrRuns, results, jobs )
  end
  @elapsed for i in 1:1:Threads.nthreads()    
    res[i,:] = take!(results);
  end
  pRaw = sum( res, dims=1 );
  return pRaw ./ sum( pRaw );
end

function bootstrapPosteriorEstimation( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )
  len      = size( errMat )[1];
  width    = size( errMat )[2];
  lenCache = samplingBatchSize;
  res      = zeros( Int64, width );
  for i in 1:1:nrRuns
    samplingIndexCache = rand( 1:size( errMat, 1 ), lenCache ) ;
    label              = argmin( mean( errMat[samplingIndexCache,:], dims=1 )[1,:] );
    res[label]        += 1; 
  end
  return  res ./ nrRuns; 
end

function bootstrapPosteriorEstimation!( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64, p::Array{Float64} )
  pB = bootstrapPosteriorEstimation( Matrix( errMat ), sampleSize, nrRuns );
  for (i,val) in enumerate( pB )
    p[i] = val;
  end 
end

tasks    = Vector{Task}( undef, Threads.nthreads() );
p        = zeros( Float64, 16 );
pMat     = zeros( Float64, 4, 16 );
a()      = bootstrapPosteriorEstimation!( errMat, sampleSize, nrRuns, p );
a()      = bootstrapPosteriorEstimation!( errMat, sampleSize, nrRuns, res[1] );
tasks[1] = Task( a );
istaskstarted( tasks[1] );
schedule( tasks[1] );
yield();
istaskdone( tasks[1] );


function v2( errMat::Matrix{Float64}, samplingBatchSize::Int64, nrRuns::Int64 )
  tasks = Vector{Task}( undef, Threads.nthreads() );
  width = size( errMat, 2 );
  res   = Vector{ Vector{Float64} }( undef, Threads.nthreads() );
  p     = zeros( Float64, width );
  for i=1:1:Threads.nthreads()
    res[i]   = zeros( Float64, width );
    a()      = bootstrapPosteriorEstimation!( errMat, sampleSize, nrRuns, res[i] );
    tasks[i] = Task( a );
    schedule( tasks[i] );
    yield();
  end
  for i=1:1:Threads.nthreads()
  end
  yield();
  for vec in res
    p .+= vec;
  end
  return p ./ Threads.nthreads();
end





