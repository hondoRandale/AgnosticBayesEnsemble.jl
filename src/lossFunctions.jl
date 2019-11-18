
  """
  ## @param:  y  Float64 - prediction
  ## @param:  t  Float64 - ground truth label
  ## @brief:  compute hinge loss for a single binary classification
  ## @return: Float64 hinge error
  """
  function hingeLoss( y::Float64, t::Float64 )
    e = 1.0 - t * y;
    if e > 0.0
      return e;
    else
      return 0.0;
    end  
  end

  """
  ## @param:  y  Vector{Float64} - predictions
  ## @param:  t  Vector{Float64} - ground truth labels
  ## @brief:  compute hinge loss for a single binary classification
  ## @return: Vector{Float64}  hinge errors
  """
  function hingeLoss( y::Vector{Float64}, t::Vector{Float64} )
    return hingeLoss.( y, t );
  end  

  """
  ## @param:  y  Float64 - prediction
  ## @param:  t  Float64 - ground truth label
  ## @brief:  compute mean squared error for a single binary classification
  ## @return: Vector{Float64}  hinge errors
  """
  function MSE( y::Float64, t::Float64 )
    return ( y - t )^2;
  end
  
  """
  ## @param:  y  Vector{Float64} - predictions
  ## @param:  t  Vector{Float64} - ground truth labels
  ## @brief:  compute mean squared error for a vector of classifications
  ## @return: Vector{Float64}  mse errors
  """
  function MSE( Y::Vector{Float64}, T::Vector{Float64} )
    return MSE.( Y, T );
  end
  
  """
  ## @param:  y  Float64 - prediction
  ## @param:  t  Float64 - ground truth label
  ## @brief:  compute zero-one loss for a single binary classification
  ## @return: Vector{Float64}  zero-one errors
  """
  function zeroOneLoss( y::Float64, t::Float64 )
   if y ≈ t
     return 0.0; 
   else
     return 1.0; 
   end       
  end
  
  """
  ## @param:  y  Vector{Float64} - predictions
  ## @param:  t  Vector{Float64} - ground truth labels
  ## @brief:  compute mean zero-one loss for a vector of classifications
  ## @return: Vector{Float64}  mse errors
  """
  function zeroOneLoss( Y::Vector{Float64}, T::Vector{Float64} )
    return zeroOneLoss.( Y, T );
  end