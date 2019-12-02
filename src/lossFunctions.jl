module lossFunctions
  export hingeLoss, MSE, zeroOneLoss

  """
      function hingeLoss( y::Float64, t::Float64 )




      compute hinge loss for a single binary classification.
      #Arguments
        - `y::Float64`: predicted value.    
        - `t::Float64`: ground truth value.
      #Return
        - `Float64`:    hingeLoss error.
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
      hingeLoss( y::Vector{Float64}, t::Vector{Float64} )




      compute hinge loss for a vector of binary classifications.
      #Arguments
        - `Vector{Float64}`: predicted values.    
        - `Vector{Float64}`: ground truth values.
      #Return
        - `Vector{Float64}`: vector of hingeLoss errors.
  """
  function hingeLoss( y::Vector{Float64}, t::Vector{Float64} )
    return hingeLoss.( y, t );
  end  

  """
      MSE( y::Float64, t::Float64 )




      compute mean squared error for a single binary classification.
        #Arguments
        - `Float64`: predicted value.    
        - `Float64`: ground truth value.
      #Return
        - `Float64`: hingeLoss error.
  """
  function MSE( y::Float64, t::Float64 )
    return ( y - t )^2;
  end
  
  """
      MSE( Y::Vector{Float64}, T::Vector{Float64} )




      compute mean squared error for a vector of binary classifications.
        #Arguments
        - `Vector{Float64}`: predicted value.    
        - `Vector{Float64}`: ground truth value.
      #Return
        - `Vector{Float64}`: vector of hingeLoss errors.
  """
  function MSE( Y::Vector{Float64}, T::Vector{Float64} )
    return MSE.( Y, T );
  end
  
    """
        zeroOneLoss( y::Float64, t::Float64 )




        compute zero-one loss for a vector for a binary classifications.      
          #Arguments
          - `Float64`: predicted value.    
          - `Float64`: ground truth value.
          #Return
          - `Float64`: zero-one loss error.

  """
  function zeroOneLoss( y::Float64, t::Float64 )
   if y ≈ t
     return 0.0; 
   else
     return 1.0; 
   end       
  end

  
  """
        zeroOneLoss( y::Float64, t::Float64 )




        compute zero-one loss for a vector of binary classifications.      
          #Arguments
          - `Vector{Float64}`: predicted values.    
          - `Vector{Float64}`: ground truth values.
          #Return
          - `Vector{Float64}`: vector of zero-one loss errors.

  """
  function zeroOneLoss( Y::Vector{Float64}, T::Vector{Float64} )
    return zeroOneLoss.( Y, T );
  end
end