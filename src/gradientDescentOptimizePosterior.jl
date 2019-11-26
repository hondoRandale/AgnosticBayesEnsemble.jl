using DataFrames
using Distributions
using GLM
using LineSearches
using Optim
using Random
using Statistics
using ProgressMeter

  """
  ## @param:  d  Int64  number of outcomes
  ## @brief:  draw a random distributiob over d outputs
  ## @return: Vector{Float64} p with sum(p) == 1.0
  """
  function randPosterior( d::Int64 )
    p = rand( collect(0.0:1e-3:10000.0), d );
    return p ./ sum(p);
  end

  """
  ## @param:  Y  Matrix{Float64}
  ## @param:  t  Vector{Float64}
  ## @brief:  compute the rel. percantage each prediction represents the true function.
  ## @return: Vector{Float64}
  """
  function posteriorLinearBasis( Y::Matrix{Float64}, t::Vector{Float64} ) 
    return ( transpose( Y ) * t ) ./ ( transpose(t) * t )
  end

  """
  ## @param:  OutputVector - YOut::Vector{Float64} hopfield encoded label vector and return value
  ## @param:  InputVector  - YIn::Vector{Float64}  {0,1} encoded binary label vector
  ## @brief:  transform label encoding from {0,1} to {-1,1}
  ## @return: nothing
  """
  function toHopfieldEncoding!( YOut::Vector{Float64}, YIn::Vector{Float64} )
    index0 = YIn .== 0;
    for i in 1:1:size( YIn, 1 )
      if index0[i] 
        YOut[i] = -1.0;
      else
        YOut[i] = 1.0     
      end  
    end
  end

  """
  ## @param:  OutputMatrix - YOut::Matrix{Float64} hopfield encoded label vector and return value
  ## @param:  InputMatrix  - YIn::Matrix{Float64}  {0,1} encoded binary label vector
  ## @brief:  transform label encoding from {0,1} to {-1,1}
  ## @return: nothing
  """
  function toHopfieldEncoding!( YOut::Matrix{Float64}, YIn::Matrix{Float64} )
    for row in 1:1:size( YIn, 1 )
      for col in 1:1:size( YIn, 2 )
        if YIn[row,col] == 0.0
          YOut[row,col] = -1.0;
        end   
      end  
    end
  end

  """
  ## @param:  cache     Vector{Float64}  resulting gradient
  ## @param:  posterior Vector{Float64}  starting solution
  ## @param:  predMat   Matrix{Float64}  prediction Matrix -one column foreach prediction model-
  ## @param:  T         Vector{Float64}  ground truth label vector
  ## @brief:  compute gradient under hinge loss function
  ## @return: nothing
  """
  function gradientHinge!( cache::Vector{Float64}, posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64} )
    width  = size( predMat, 2 );
    height = size( predMat, 1 );
    tmp    = -mean( predMat .* repeat( T, outer = [ 1, width ] ), dims=1 );
    for i in 1:1:width
      cache[i] = tmp[i];
    end    
  end

  """
  ## @param:  cache     Vector{Float64}  resulting gradient
  ## @param:  posterior Vector{Float64}  starting solution
  ## @param:  predMat   Matrix{Float64}  prediction Matrix - one column foreach prediction model -
  ## @param:  T         Vector{Float64}  ground truth label vector
  ## @param:  α         Float64          regularization param 1
  ## @param:  β         Float64          regularization param 2
  ## @param:  entTarget Float64          regularization param 3
  ## @brief:  compute gradient under hinge loss function regularized
  ## @return: nothing
  """
  function gradientHingeRegularized!( cache::Vector{Float64}, posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64}, α::Float64, β::Float64, entTarget::Float64 )
    width  = size( predMat, 2 );
    height = size( predMat, 1 );
    δHinge = -median( predMat .* repeat( T, outer = [ 1, width ] ), dims=1 );
    δHinge = δHinge .+ α * 2 * ( sum( posterior ) - 1.0 ) * ones( Float64, width );     
    δHinge = δHinge .+ β * 2 * ( entropy( posterior ) - entTarget ) * ( ones( Float64, width ) .+ log.( posterior ) ); 
    for i in 1:1:width
      cache[i] = δHinge[i];
    end
  end

  """
  ## @param:  cache     Vector{Float64}  resulting gradient
  ## @param:  posterior Vector{Float64}  starting solution
  ## @param:  predMat   Matrix{Float64}  prediction Matrix - one column foreach prediction model -
  ## @param:  T         Vector{Float64}  ground truth label vector
  ## @brief:  compute gradient under MSE function
  ## @return: nothing
  """
  function gradientMSE!( cache::Vector{Float64}, posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64} )
    width  = size( predMat, 2 );
    height = size( predMat, 1 );  
    yEns   = predMat * posterior;
    tmp    = 2 * mean( repeat( ( yEns .- T ), outer = [ 1, width ] ) .* predMat, dims=1 );
    for i in 1:1:width
      cache[i] = tmp[i];
    end
  end

  """
  ## @param:  cache     Vector{Float64}  resulting gradient
  ## @param:  posterior Vector{Float64}  starting solution
  ## @param:  predMat   Matrix{Float64}  prediction Matrix - one column foreach prediction model -
  ## @param:  T         Vector{Float64}  ground truth label vector
  ## @param:  α         Float64          regularization param 1
  ## @param:  β         Float64          regularization param 2
  ## @param:  entTarget Float64          regularization param 3
  ## @brief:  compute gradient under hinge MSE function regularized
  ## @return: nothing
  """
  function gradientMSERegularized!( cache::Vector{Float64}, posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64}, α::Float64, β::Float64, entTarget::Float64 )
    width  = size( predMat, 2 );
    height = size( predMat, 1 );  
    yEns   = predMat * posterior;
    δMSE   = 2 * median( repeat( ( yEns .- T ), outer = [ 1, width ] ) .* predMat, dims=1 )[1,:];
    δMSE   = δMSE .+ α * 2 * ( sum( posterior ) - 1.0 ) * ones( Float64, width );     
    δMSE   = δMSE .+ β * 2 * ( entropy( posterior ) - entTarget ) * ( ones( Float64, width ) + log.( posterior ) ); 
    for i in 1:1:width
      cache[i] = δMSE[i];
    end
  end

  """
  ## @param: posterior Vector{Float64}  starting solution
  ## @param: predMat   Matrix{Float64}  prediction Matrix - one column foreach prediction model -
  ## @param: T         Vector{Float64}  ground truth label vector
  ## @param: max_iter  Int64            max. number of gradient iterations
  ## @brief: optmize initial solution posterior using gradient descend
  ## @return:
  """
  function δOptimizationHinge( posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64}, max_iter::Int64 )
    d               = size( predMat, 2 );  
    lower           = zeros( Float64, d );
    upper           = ones( Float64,  d );
    inner_optimizer = Optim.GradientDescent()
    function g!( cache::Vector{Float64}, posterior::Vector{Float64} )
      gradientHinge!( cache, posterior, predMat, T );
    end
    function f( posterior )
      return mean( lossFunctions.hingeLoss.( predMat * posterior, T ) )
    end
    results = Optim.optimize( f, g!, lower, upper, posterior, Optim.Fminbox( inner_optimizer ) )
  end    

  """
  ## @param:  posterior Vector{Float64}  starting solution
  ## @param:  predMat   Matrix{Float64}  prediction Matrix - one column foreach prediction model -
  ## @param:  T         Vector{Float64}  ground truth label vector
  ## @param:  max_iter  Int64            max. number of gradient iterations
  ## @param:  α         Float64          regularization param 1
  ## @param:  β         Float64          regularization param 2
  ## @param:  entTarget Float64          regularization param 3
  ## @brief:
  ## @return:
  """
  function δOptimizationHingeRegularized( posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64}, max_iter::Int64, α::Float64, β::Float64, entTarget::Float64 )
    d               = size( predMat, 2 );  
    lower           = zeros( Float64, d );
    upper           = ones( Float64,  d );
    inner_optimizer = Optim.GradientDescent( linesearch=LineSearches.Static() );
    function g!( cache::Vector{Float64}, posterior::Vector{Float64} )
      gradientHingeRegularized!( cache, posterior, predMat, T, α, β, entTarget );
    end
    function f( posterior )
      return mean( lossFunctions.hingeLoss.( predMat * posterior, T ) )
    end
    results = Optim.optimize( f, g!, lower, upper, posterior, Optim.Fminbox( inner_optimizer ) )
  end

  """
  ## @param:  posterior Vector{Float64}  starting solution
  ## @param:  predMat   Matrix{Float64}  prediction Matrix - one column foreach prediction model -
  ## @param:  T         Vector{Float64}  ground truth label vector
  ## @param:  max_iter  Int64            max. number of gradient iterations
  ## @return: 
  """
  function δOptimizationMSE( posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64}, max_iter::Int64 )
    d               = size( predMat, 2 );  
    lower           = zeros( Float64, d );
    upper           = ones( Float64,  d );
    inner_optimizer = Optim.GradientDescent( linesearch=LineSearches.Static() )
    function g!( cache::Vector{Float64}, posterior::Vector{Float64} )
      gradientMSE!( cache, posterior, predMat, T );
    end
    function f( posterior )
      return mean( lossFunctions.MSE.( predMat * posterior, T ) )
    end
    results = Optim.optimize( f, g!, lower, upper, posterior, Optim.Fminbox( inner_optimizer ) );
    return results
  end   

  """
  ## @param:  posterior Vector{Float64}  starting solution
  ## @param:  predMat   Matrix{Float64}  prediction Matrix - one column foreach prediction model -
  ## @param:  T         Vector{Float64}  ground truth label vector
  ## @param:  max_iter  Int64            max. number of gradient iterations
  ## @param:  α         Float64          regularization param 1
  ## @param:  β         Float64          regularization param 2
  ## @param:  entTarget Float64          regularization param 3
  ## @brief:  
  ## @return: nothing
  """
  function δOptimizationMSERegularized( posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64}, max_iter::Int64, α::Float64, β::Float64, entTarget::Float64 )
    d               = size( predMat, 2 );  
    lower           = zeros( Float64, d );
    upper           = ones( Float64,  d );
    inner_optimizer = Optim.GradientDescent( linesearch=LineSearches.Static() );
    function g!( cache::Vector{Float64}, posterior::Vector{Float64} )
      gradientMSERegularized!( cache, posterior, predMat, T, α, β, entTarget );
    end
    function f( posterior )
      return mean( lossFunctions.MSE.( predMat * posterior, T ) )
    end
    results = Optim.optimize( f, g!, lower, upper, posterior, Optim.Fminbox( inner_optimizer ) )
  end   

  """
  ## @param:  posterior       Vector{Float64}        starting solution
  ## @param:  predMat         Matrix{Float64}        prediction Matrix - one column foreach prediction model -
  ## @param:  T               Vector{Float64}        ground truth label vector
  ## @param:  nrRunsRange     Tuple{Float64,Float64} parameter range
  ## @param:  αRange          Tuple{Float64,Float64} parameter range
  ## @param:  βRange          Tuple{Float64,Float64} parameter range
  ## @param:  relEntropyRange Tuple{Float64,Float64} parameter range
  ## @param:  generations     Int64                  numbers of generations to spawn
  ## @param:  siblings        Int64                  numbers of siblings per spawn
  ## @brief:  search for best regularization params using Mean Squared Error
  ## @return:
  """
  function δTuneMSEMeta(;posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64}, nrRunsRange::Tuple{Float64,Float64}, αRange::Tuple{Float64,Float64}, βRange::Tuple{Float64,Float64}, relEntropyRange::Tuple{Float64,Float64}, generations::Int64, siblings::Int64 )
    nrPredRun       = Int64( 1e4 );
    d               = size( predMat, 2 );
    height          = size( predMat, 1 );
    entropyMax      = -log( 1 / d );
    stepNrRuns      = ( nrRunsRange[2] - nrRunsRange[1] ) / 1e5;
    stepα           = ( αRange[2] - αRange[1] ) / 1e5;
    stepβ           = ( βRange[2] - βRange[1] ) / 1e5;
    stepEntropy     = ( relEntropyRange[2] * entropyMax - relEntropyRange[1] * entropyMax ) / 1e5;

    valRangeNrRuns  = collect( nrRunsRange[1]:stepNrRuns:nrRunsRange[2] );
    valRangeA       = collect( αRange[1]:stepα:αRange[2] );
    valRangeB       = collect( βRange[1]:stepβ:βRange[2] );
    valRangeEntropy = collect( relEntropyRange[1] * entropyMax :stepEntropy:relEntropyRange[2] * entropyMax  );
    resultDF        = DataFrame( nrRuns=zeros( Int64, siblings * generations ), α=zeros( Float64, siblings * generations ), β=zeros( Float64, siblings * generations ),
                                 Entropy=zeros( Float64, siblings * generations ), MSE=zeros( Float64, siblings * generations ) );
    parameterDF     = DataFrame( nrRuns=round.( Int64, rand( valRangeNrRuns, siblings ) ), α=rand( valRangeA, siblings ), β=rand( valRangeB, siblings ), Entropy=rand( valRangeEntropy, siblings )  );
    cnames          = string.( "w", collect( 1:1:d ) );
    append!( cnames, ["Entropy","MSE"] );
    parameterEvalDf = DataFrame( Matrix{Float64}( undef, siblings * generations, d + 2 ) );
    names!( parameterEvalDf, Symbol.( cnames ) ); ## FIXME: broke
    
    for i in 1:1:generations
      ## run  all parameters of parmeterDF, store results in resultDF, parameterEvalDf
      @showprogress 1 "Computing..." for row in 1:1:siblings
        inRow                   = ( ( i - 1 ) * siblings ) + row;
        resObject               = δOptimizationMSERegularized( posterior, predMat, T, parameterDF[row,:nrRuns], parameterDF[row,:α], parameterDF[row,:β], parameterDF[row,:Entropy] );
        pRefinedExt             = Optim.minimizer( resObject );
        performance             = mean( lossFunctions.MSE.( predMat * pRefinedExt, T ) );
        resultDF[inRow,:]        .= ( parameterDF[row,:nrRuns], parameterDF[row,:α], parameterDF[row,:β], parameterDF[row,:Entropy], performance );
        push!( pRefinedExt, entropy( pRefinedExt ), performance );
        parameterEvalDf[inRow,:] .= pRefinedExt;
      end
      lim = i*siblings;
      ## train GLM model on parameterEvalDf predict MSE
      modelGLM   = glm( @formula( MSE ~ nrRuns + α  + β + Entropy ), resultDF[1:lim,:], Binomial(), IdentityLink(), maxiter=1000 );
      ## draw new candidates
      candidates = DataFrame( nrRuns=round.( Int64, rand( valRangeNrRuns, nrPredRun ) ), α=rand( valRangeA, nrPredRun ), 
                                             β=rand( valRangeB, nrPredRun ), Entropy=rand( valRangeEntropy, nrPredRun ) );
      ## predict parmeterDF with GlM model
      MSEGLM         = convert( Array{Float64,1}, GLM.predict( modelGLM, candidates ) );
      candidates.MSE = MSEGLM;
      sort!( candidates, [:MSE] );
      parameterDF   .= candidates[1:siblings,1:4];
    end
    return resultDF, parameterEvalDf
  end  
  
  """
  ## @param:  posterior       Vector{Float64}        starting solution
  ## @param:  predMat         Matrix{Float64}        prediction Matrix - one column foreach prediction model -
  ## @param:  T               Vector{Float64}        ground truth label vector
  ## @param:  nrRunsRange     Tuple{Float64,Float64} parameter range
  ## @param:  αRange          Tuple{Float64,Float64} parameter range
  ## @param:  βRange          Tuple{Float64,Float64} parameter range
  ## @param:  relEntropyRange Tuple{Float64,Float64} parameter range
  ## @param:  generations     Int64                  numbers of generations to spawn
  ## @param:  siblings        Int64                  numbers of siblings per spawn
  ## @brief:  search for best regularization params using hingeLoss
  ## @return: 
  """
  function δTuneHingeMeta(;posterior::Vector{Float64}, predMat::Matrix{Float64}, T::Vector{Float64}, nrRunsRange::Tuple{Float64,Float64}, αRange::Tuple{Float64,Float64}, βRange::Tuple{Float64,Float64}, relEntropyRange::Tuple{Float64,Float64}, generations::Int64, siblings::Int64 )
    nrPredRun       = Int64( 1e5 );
    d               = size( predMat, 2 );
    height          = size( predMat, 1 );
    entropyMax      = -log( 1 / d );
    stepNrRuns      = ( nrRunsRange[2] - nrRunsRange[1] ) / 1e5;
    stepα           = ( αRange[2] - αRange[1] ) / 1e5;
    stepβ           = ( βRange[2] - βRange[1] ) / 1e5;
    stepEntropy     = ( relEntropyRange[2] * entropyMax - relEntropyRange[1] * entropyMax ) / 1e5;

    valRangeNrRuns  = collect( nrRunsRange[1]:stepNrRuns:nrRunsRange[2] );
    valRangeA       = collect( αRange[1]:stepα:αRange[2] );
    valRangeB       = collect( βRange[1]:stepβ:βRange[2] );
    valRangeEntropy = collect( relEntropyRange[1] * entropyMax :stepEntropy:relEntropyRange[2] * entropyMax  );
    resultDF        = DataFrame( nrRuns=zeros( Int64, siblings * generations ), α=zeros( Float64, siblings * generations ), β=zeros( Float64, siblings * generations ),
                                 Entropy=zeros( Float64, siblings * generations ), hLoss=zeros( Float64, siblings * generations ) );
    parameterDF     = DataFrame( nrRuns=round.( Int64, rand( valRangeNrRuns, siblings ) ), α=rand( valRangeA, siblings ), β=rand( valRangeB, siblings ), Entropy=rand( valRangeEntropy, siblings )  );
    cnames          = string.( "w", collect( 1:1:d ) );
    append!( cnames, ["Entropy","hLoss"] );
    parameterEvalDf = DataFrame( Matrix{Float64}( undef, siblings * generations, d + 2 ) );
    names!( parameterEvalDf, Symbol.( cnames ) ); ## FIXME: broke
    
    for i in 1:1:generations
      ## run  all parameters of parmeterDF, store results in resultDF, parameterEvalDf
      @showprogress 1 "Computing..." for row in 1:1:siblings
        inRow                   = ( ( i - 1 ) * siblings ) + row;
        resObject               = δOptimizationHingeRegularized( posterior, predMat, T, parameterDF[row,:nrRuns], parameterDF[row,:α], parameterDF[row,:β], parameterDF[row,:Entropy] );
        pRefinedExt             = Optim.minimizer( resObject );
        performance             = mean( lossFunctions.hingeLoss.( predMat * pRefinedExt, T ) );
        resultDF[inRow,:]        .= ( parameterDF[row,:nrRuns], parameterDF[row,:α], parameterDF[row,:β], parameterDF[row,:Entropy], performance );
        push!( pRefinedExt, entropy( pRefinedExt ), performance );
        parameterEvalDf[inRow,:] .= pRefinedExt;
      end
      lim = i*siblings;
      ## train GLM model on parameterEvalDf predict MSE
      modelGLM   = glm( @formula( hLoss ~ nrRuns + α  + β + Entropy ), resultDF[1:lim,:], Normal(), IdentityLink(), maxiter=1000 );
      ## draw new candidates
      candidates = DataFrame( nrRuns=round.( Int64, rand( valRangeNrRuns, nrPredRun ) ), α=rand( valRangeA, nrPredRun ), 
                                             β=rand( valRangeB, nrPredRun ), Entropy=rand( valRangeEntropy, nrPredRun ) );
      ## predict parmeterDF with GlM model
      hLossGLM         = convert( Array{Float64,1}, GLM.predict( modelGLM, candidates ) );
      candidates.hLoss = hLossGLM;
      sort!( candidates, [:hLoss] );
      parameterDF   .= candidates[1:siblings,1:4];
    end
    return resultDF, parameterEvalDf
  end 

  
