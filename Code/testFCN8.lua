-- This file tests if our FCN8 network can work 
require 'paths'
paths.dofile('FCN8.lua')
testData = torch.rand(3, 200, 260)
print(#testData)

predicted = fcn_net:forward(testData)
print(#predicted)



