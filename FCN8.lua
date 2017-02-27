require 'nn'

local fcn_net = nn.Sequential()

-- building block
local function ConvReLU(nInputPlane, nOutputPlane)
  fcn_net:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,3,1,1,1,1))
  -- 3x3 convolution kernel, stride of 1, pad of 1
  fcn_net:add(nn.ReLU(true))
  return fcn_net
end

-- conv1_1 & relu1_1
fcn_net:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,64,1,1,100,100))
fcn_net:add(nn.ReLU(true))
-- conv1_2 & relu1_1
ConvReLU(64,64)
-- pool1
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv2_1 & relu2_1
ConvReLU(64,128)
-- conv2_2 & relu2_2
ConvReLU(128,128)
-- pool2
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv3_1 && relu3_1
ConvReLU(256,256)
-- conv3_2 && relu3_2
ConvReLU(256,256)
-- conv3_3 && relu3_3
ConvReLU(256,256)

fcn_8s = nn.pa

-- pool3 
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv4_1 & relu4_1
ConvReLU(256,512)
-- conv4_2 & relu4_2
ConvReLU(512,512)
-- conv4_3 & relu4_3
ConvReLU(512,512)
-- pool4
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv5_1 & relu5_1
ConvReLU(512,512)
-- conv5_2 & relu5_2
ConvReLU(512,512)
-- conv5_3 & relu5_3
ConvReLU(512,512)
-- pool5
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- fc6
fcn_net:add(nn.SpatialConvolution(512,4096,7,7,1,1,0,0))
-- relu6
fcn_net:add(nn.ReLU(true))
-- drop6
fcn_net:add(nn.Dropout(0.5))
-- fc7
fcn_net:add(nn.SpatialConvolution(4096,4096,1,1,1,1,0,0))
-- drop7
fcn_net:add(nn.Dropout(0.5))
-- score_fr
fcn_net:add(nn.SpatialConvolution(4096,21,1,1,1,1,0,0))
-- upscore2
fcn_net:add(nn.SpatialFullConvolution(21,21,4,4,2,2,0,0,0,0):noBias())

--score_pool4
fcn_net:add(nn.SpatialConvolution(512,21,1,1,1,1,0,0))
--score_pool4c(upscore2)
fcn_net:add(nn.)
--fuse_pool4(fuse_pool4)
fcn_net:add
--upscore_l4(upscore_pool4)
fcn_net:add(nn.SpatialFullConvolution(21,21,4,4,2,2,0,0,0,0):noBias())
--score_pool3
fcn_net:add(nn.SpatialConvolution(256, 21, 1, 1, 1, 1, 0, 0))
--score_pool3c
--fuse_pool3
--upscore8
--score
--loss


vgg:add(nn.View(512))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,10))
vgg:add(classifier)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
