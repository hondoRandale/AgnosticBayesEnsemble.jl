using DataFrames


function distortBinaryPrediction( y::BitArray{1}, distortionFactor::Float64 )
    res          = deepcopy( y );   
    indices      = rand( 1:1:size( y, 1 ), round( Int64, distortionFactor * size( y, 1 ) ) );
    res[indices] = .!y[indices];
    return res;
end   

function makeupPredictions()
  n    = 100000;
  y    = Bool.( rand( 0:1,n ) );
  yH1  = Float64.( distortBinaryPrediction( y, 0.20 ) );
  yH2  = Float64.( distortBinaryPrediction( y, 0.21 ) );
  yH3  = Float64.( distortBinaryPrediction( y, 0.22 ) );
  yH4  = Float64.( distortBinaryPrediction( y, 0.23 ) );
  yH5  = Float64.( distortBinaryPrediction( y, 0.24 ) );
  yH6  = Float64.( distortBinaryPrediction( y, 0.24 ) );
  yH7  = Float64.( distortBinaryPrediction( y, 0.26 ) );
  yH8  = Float64.( distortBinaryPrediction( y, 0.27 ) );
  yH9  = Float64.( distortBinaryPrediction( y, 0.28 ) );
  yH10 = Float64.( distortBinaryPrediction( y, 0.29 ) );
  yH11 = Float64.( distortBinaryPrediction( y, 0.30 ) );
  yH12 = Float64.( distortBinaryPrediction( y, 0.33 ) );
  yH13 = Float64.( distortBinaryPrediction( y, 0.34 ) );
  yH14 = Float64.( distortBinaryPrediction( y, 0.35 ) );
  yH15 = Float64.( distortBinaryPrediction( y, 0.36 ) );
  yH16 = Float64.( distortBinaryPrediction( y, 0.37 ) );
  y    = Float64.( y );
  limit           = round( Int64, 0.7 * size( y, 1 ) ); 
  predictions     = hcat( yH1, yH2, yH3, yH4, yH5, yH6, yH7, yH8, yH9, yH10, yH11, yH12, yH13, yH14, yH15, yH16 );
  predTraining    = predictions[1:limit,:];
  predEval        = predictions[(limit+1):end,:];
  predMatTraining = convert( Matrix, predTraining );
  predMatEval     = convert( Matrix, predEval );
  yTraining       = Float64.( y[1:limit] );
  yEval           = Float64.( y[(limit+1):end] );
  errMatTraining  = ( repeat(  Float64.( yTraining ),outer = [1,size(predMatTraining,2)] ) .- predMatTraining ).^2;
  return predMatTraining, predMatEval, yTraining, yEval, errMatTraining;
end
