module argmaxUniProb  
  export argmaxProb
  using Statistics

  """
      argmaxProb( r::Matrix{Float64} )




      compute maximum value per row, if several cols hold the maximum value, randomly choose among them.
      #Arguments
        - `r::Matrix{Float64}`: risk matrix each column represents one hypothesis.
      #Return
        - `Vector{Float64}`:    indices of maximal values.
  """
  function argmaxProb( r::Matrix{Float64} )
    index = Vector{Int64}( undef, size( r, 1 ) );
    for i in size( index, 1 )
      index[i] = argmaxProb( r[i,:] );
    end
    return index; 
  end

  """
      argmaxProb!( r::Matrix{Float64}, v::Vector{Int64} )




      compute maximum value per row, if several cols have maximum value randomly choose among them.
      #Arguments
        - `r::Matrix{Float64}`: risk matrix each column represents one hypothesis.
        - `v::Vector{Int64} `:  resulting vector.
      #Return
        - `nothing`:            nothing.
  """
  function argmaxProb!( r::Matrix{Float64}, v::Vector{Int64} )
    @assert size( v,1 ) == size( r, 2 );
    for i=1:size( v, 1 )
     v[i] = argmaxProb( r[i,:] );
    end
  end
  
  """
      argmaxProb( values::Vector{Float64} )



      compute minimum, if several indices have maximum value randomly choose among them.
      #Arguments
        - `values::Vector{Float64}`: values to compute argmax on.
      #Return
        - `Vector{Float64}`:         indices of maximal values.  
  """
  function argmaxProb( values::Vector{Float64} )
    maxPositions = collect( 1:1:size( values, 1 ) )[ values .== maximum( values ) ];
    if size( maxPositions, 1 ) > 1
      res = maxPositions[ rand( 1:size( maxPositions, 1 ) ) ];  
    else
      res = maxPositions[1];    
    end
    return res;
  end
end  