-- This file tests if our FCN8 network can work 
require 'paths'
require 'image'
require 'cutorch'
paths.dofile('FCN8.lua')
print(fcn_net)
testData = torch.rand(3, 186, 186)
testData = testData:cuda()
print(#testData)

predicted = fcn_net:forward(testData)
print(#predicted)

-- Liang's directory
-- test_img_file = './VOC2011/JPEGImages/2007_000032.jpg'
-- test_img_file = './VOC2011/JPEGImages/2007_000033.jpg'
-- Ray's directory
test_img_file = '../../Dataset/TrainVal/VOC2011/JPEGImages/2007_000033.jpg'
testImage = image.load(test_img_file, 3, 'byte')
testImage = testImage:cuda()
print('----------------------------------------------------')
print(#testImage)
predictedLabel = fcn_net:forward(testImage)
print(#predictedLabel)
-- print(#predicted[1])
-- print(#predicted[2])



