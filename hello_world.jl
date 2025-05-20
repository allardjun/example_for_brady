
using JLD2

println("Hello from Julia!")

X = rand(100)

@save "bunch_of_rands.jld2" X
