using Statistics

function argminProb( r::Matrix{Float64} )
  index = Vector{Int64}( undef, size( r, 1 ) );
  for i in size( index, 1 )
    argminProb( r[:,i] );
  end
  return index; 
end

function argminProb( r::Matrix{Float64}, v::Array{Int64,1} )
  @assert size( v,1 ) == size( r, 2 );
  for i=1:size( v, 1 )
    v[i] = argminProb( r[:,i] );
  end
end

function argminProb( values::Vector{Float64} )
  minPositions = collect( 1:1:size( values, 1 ) )[ values .== minimum( values ) ];
  if size( minPositions, 1 ) > 1
    res = minPositions[ rand( 1:size( minPositions, 1 ) ) ];  
  else
    res = minPositions[1];    
  end
  return res;
end