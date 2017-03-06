require 'paths'
-- load the data
paths.dofile('load.lua')

print(trainset:size())
print('--------------------')
print(#trainset[1][1])
print(#trainset[1][2])

trainset:shuffle()
print('--------------------')
print(#trainset[1][1])
print(#trainset[1][2])

trainset:shuffle()
print('--------------------')
print(#trainset[1][1])
print(#trainset[1][2])
