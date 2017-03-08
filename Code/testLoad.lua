require 'paths'
-- load the data
paths.dofile('load.lua')
print(trainset:size())

for i = 1, 3 do
print('--------------------')
print('trainset:')
print(#trainset[1][1])
print(#trainset[1][2])
print(trainset[1][2])
print('valset:')
print(#valset[i][1])
print(#valset[i][2])
print('testset:')
print(#testset[i])
trainset:shuffle()
end
