module argminUniProb
  export argminProb
  using Statistics

  """
  @param:  r Matrix containing error per hypothesis and sample
  @brief:  compute minimum value per row, if several cols have minimum value randomly choose among them.
  @return: vector of minimum index per row
  """
  function argminProb( r::Matrix{Float64} )
    index = Vector{Int64}( undef, size( r, 1 ) );
    for i in size( index, 1 )
      argminProb( r[:,i] );
    end
    return index; 
  end

  """
  @param:  r Matrix{Float64} containing error per hypothesis and sample
  @param:  v Array{Int64,1}  resulting vector
  @brief:  compute minimum value per row, if several cols have minimum value randomly choose among them.
  @return: nothing
  """
  function argminProb( r::Matrix{Float64}, v::Array{Int64,1} )
    @assert size( v,1 ) == size( r, 2 );
    for i=1:size( v, 1 )
      v[i] = argminProb( r[:,i] );
    end
  end

  """
  @param:  values  Vector{Float64}  values to compute argmin on
  @brief:  compute minimum, if several indices have minimum value randomly choose among them.
  @return: index of minimum Int64
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