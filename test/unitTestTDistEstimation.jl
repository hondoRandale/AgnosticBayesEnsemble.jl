using DataFrames
using Random
using Statistics
using Test
using Optim
using MultivariateStats

println( "running T-Distribution algorithm unit tests" );

predMatTraining, predMatEval, yTraining, yEval, errMatTraining = makeupPredictions();