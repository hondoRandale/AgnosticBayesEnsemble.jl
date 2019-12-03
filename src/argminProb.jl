module argminUniProb
  export argminProb
  using Statistics

  """
      argminProb( r::Matrix{Float64} )




      compute minimum value per row, if several cols hold the minimum value, randomly choose among them.
      #Arguments
        - `r::Matrix{Float64}`: risk matrix each column represents one hypothesis.
      #Return
        - `Vector{Float64}`:    indices of maximal values.
  """
  function argminProb( r::Matrix{Float64} )
    index = Vector{Int64}( undef, size( r, 1 ) );
    for i in size( index, 1 )
      index[i] = argminProb( r[i,:] );
    end
    return index; 
  end

  """
      argminProb!( r::Matrix{Float64}, v::Vector{Int64} )




      compute minimum value per row, if several cols have maximum value randomly choose among them.
      #Arguments
        - `r::Matrix{Float64}`: risk matrix each column represents one hypothesis.
        - `v::Vector{Int64} `:  resulting vector.
      #Return
        - `nothing`:            nothing.
  """
  function argminProb!( r::Matrix{Float64}, v::Vector{Int64} )
    @assert size( v,1 ) == size( r, 2 );
    for i=1:size( v, 1 )
      v[i] = argminProb( r[:,i] );
    end
  end

  """
      argminProb( values::Vector{Float64} )



      compute minimum, if several indices have minimum value randomly choose among them.
      #Arguments
        - `values::Vector{Float64}`: values to compute argmin on.
      #Return
        - `Vector{Float64}`:         indices of minimal values.  
  """
  function argminProb( values::Vector{Float64} )
    minPositions = collect( 1:1:size( values, 1 ) )[ values .== minimum( values ) ];
    if size( minPositions, 1 ) > 1
      res = minPositions[ rand( 1:size( minPositions, 1 ) ) ];  
    else
      res = minPositions[1];    
    end
    return res;
  end
end  