require 'nn'
require 'nngraph'
require 'paths'
require 'cunn'
require 'cudnn'

opt = {
	learningRate = 0.0001, -- learning rate 10^-4 as per the paper
	maxIteration = 5, -- max epochs
	momentum = 0.9,
	batch_size = 20
}

-- train 
function train()
	-- load net module
	paths.dofile('FCN8.lua')
	fcn_net = fcn_net:cuda()
	print(fcn_net)
	-- criterion for loss
	criterion = cudnn.SpatialCrossEntropyCriterion()
	criterion = criterion:cuda()
	-- trainset 
	trainset = 
	trainset.data = trainset.data:cuda()
	trainset.label = trainset.label:cuda()
	-- trainer
	trainer = nn.StochasticGradient(fcn_net, criterion)
	trainer.learningRate = opt.learningRate
	trainer.maxIteration = opt.maxIteration
	trainer:train(trainset) 
	-- example for dataset
	-- dataset={};
	-- function dataset:size() return 100 end -- 100 examples
	-- for i=1,dataset:size() do 
	--   local input = torch.randn(2);     -- normally distributed example in 2d
	--   local output = torch.Tensor(1);
	--   if input[1]*input[2]>0 then     -- calculate label for XOR function
	--     output[1] = -1;
	--   else
	--     output[1] = 1
	--   end
	--   dataset[i] = {input, output}
	-- end
end

-- functions to calculate all four metrics
-- pixel accuracy
local function cal_pixel_accuracy()

end
-- pixel accuracy 
local function cal_mean_accuracy()
end
-- mean IU
local function cal_iu()
end
-- frequency weighted IU
local function cal_fw_iu()
end

-- test
function test()
	testset = 

	for 1=1,10000 do
		local true_seg = 
		local net_seg = fcn_net:forward(testset.data[100])
	end

end

for i = 1,opt.max_epoch do
	train()
	test()
end