-- This file tests if our FCN32 network can work 
require 'paths'
require 'image'
paths.dofile('fcn16.lua')
print(fcn_net)
testData = torch.rand(3, 186, 186)
print(#testData)

predicted = fcn_net:forward(testData)
print(#predicted)


-- test_img_file = './VOC2011/JPEGImages/2007_000032.jpg'
test_img_file = '../VOC2011/JPEGImages/2007_000033.jpg'
testImage = image.load(test_img_file, 3, 'byte')
testImage = testImage:double()
print('----------------------------------------------------')
print(#testImage)
predictedLabel = fcn_net:forward(testImage)
print(#predictedLabel)