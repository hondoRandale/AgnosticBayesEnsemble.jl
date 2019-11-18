function distortBinaryPrediction( y::BitArray{1}, distortionFactor::Float64 )
    res          = deepcopy( y );   
    indices      = rand( 1:1:size( y, 1 ), round( Int64, distortionFactor * size( y, 1 ) ) );
    res[indices] = .!y[indices];
    return res;
end   

function makeupPredictions()
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
  y    = Float64.( y );
  limit           = round( Int64, 0.7 * size( y, 1 ) ); 
  predictions     = DataFrame( h1=yH1, h2=yH2, h3=yH3, h4=yH4, h5=yH5, h6=yH6, h7=yH7, h8=yH8, h9=yH9, h10=yH10, h11=yH11, h12=yH12, h13=yH13, h14=yH14, h15=yH15, h16=yH16 );
  predTraining    = predictions[1:limit,:];
  predEval        = predictions[(limit+1):end,:];
  predMatTraining = convert( Matrix{Float64}, predTraining );
  predMatEval     = convert( Matrix{Float64}, predEval );
  yTraining       = Float64.( y[1:limit] );
  yEval           = Float64.( y[(limit+1):end] );
  errMatTraining  = ( repeat(  Float64.( yTraining ),outer = [1,size(predMatTraining,2)] ) .- predMatTraining ).^2;
  return predMatTraining, predMatEval, yTraining, yEval, errMatTraining;
end