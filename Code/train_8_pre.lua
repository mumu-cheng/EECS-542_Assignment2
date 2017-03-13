require 'nn'
require 'nngraph'
require 'paths'
require 'cunn'
require 'cudnn'
require 'optim'
require 'cutorch'
require 'image'

-- load the data
-- paths.dofile('load.lua')
print('>>>> Start loading training dataset')
trainset = torch.load('../Datasett7/trainset.t7')

function swap(array, index1, index2)
    array[index1], array[index2] = array[index2], array[index1]
end

function trainset:shuffle()
	local counter = trainset:size()
	while counter > 1 do
		local index = math.random(counter)
		swap(trainset, index, counter)
		counter = counter - 1
	end
end

-- the member function of trainset, resize every image in this trainset by ratio
function trainset:resize(ratio)
	for i = 1, trainset:size() do
		local H = trainset[i][1]:size()[2]
		local W = trainset[i][1]:size()[3]
		local new_H = torch.floor(ratio * H)
		local new_W = torch.floor(ratio * W)
		trainset[i][1] = image.scale(trainset[i][1], new_H, new_W, "simple")
		trainset[i][2] = image.scale(trainset[i][2], new_H, new_W, "simple")
	end
end

trainset:resize(0.5)

print('>>>> Start loading validation dataset')
-- valset = torch.load('../Datasett7/valset.t7')
print('>>>> Finish loading dataset')

-- load the functions to calculate metrics 
paths.dofile('metrics.lua')
-- write the loss to a text file and read from there to plot the loss as training proceeds
logger = optim.Logger('train_loss_8_pre_261.log')
logger:setNames{'Epoch','Training loss.','Val pixel acc.','Val mean acc.','mean IU','FW IU'}

-- print('learningRate= ' .. optimState.learningRate)
-- hyperparameter
local config = {
	batch_size = 1, -- online learning
	max_epoch = 600, -- max number of epochs
	trainset_size = trainset:size(),
	-- valset_size = valset:size(),
}

-- load the untrained model
-- paths.dofile('fcn8.lua')
paths.dofile('CropTable.lua')
fcn_net = torch.load('./fcn8_pre260.t7')
fcn_net = fcn_net:cuda()
print('>>>> Finish loading net and converting net to cuda')

-- train 
function train()
	local params, gradParams = fcn_net:parameters()
	-- params = params:cuda()
	-- gradParams = gradParams:cuda()
	-- states variables for the optimization process
	optimState = {}
	for i=1, #params-3 do
	  table.insert(optimState, {
	                              learningRate = 0,
	                              learningRateDecay = 0.0,
	                              momentum = 0,
	                              dampening = 0.0,
	                              weightDecay = 0
	                            }
	              )
	end
	for i=#params-2, #params do
	  table.insert(optimState, {
	                              learningRate = 0.0000001,
	                              learningRateDecay = 0.0,
	                              momentum = 0.90,
	                              dampening = 0.0,
	                              weightDecay = 0.0005
	                            }
	              )
	end
	-- criterion for loss
	criterion = cudnn.SpatialCrossEntropyCriterion()
	criterion = criterion:cuda()
	-- start to train the net
	for epoch = 261, config.max_epoch do

		local time = sys.clock()

		print('>>>> Starting to train epoch ' .. epoch .. ':')
   		cur_loss = 0
   		for iter = 1, config.trainset_size do
   			collectgarbage()
   			fcn_net:zeroGradParameters()
			batchInputs = trainset[iter][1]:cuda()
			batchLabels = trainset[iter][2]:cuda()
			local outputs = fcn_net:forward(batchInputs)
			-- ignore_label: 255; pixels of 255 are not counted into loss function
			if torch.max(batchLabels) == 255 then
				local idx = batchLabels:eq(255)
				batchLabels[idx] = 1
				outputs[1][1][idx] = 1
				for j = 2,21 do
					outputs[1][j][idx] = 0
				end
			end
			-- calculate loss
  			local loss = criterion:forward(outputs, batchLabels)
  			local dloss_doutputs = criterion:backward(outputs, batchLabels)
  			fcn_net:backward(batchInputs, dloss_doutputs)
      		-- update parameters
      		for i = #params-2, #params do
      			local function feval()
      				return loss, gradParams[i]
      			end
	   			optim.sgd(feval, params[i], optimState[i])
	   		end
	   		cur_loss = cur_loss + loss
	   	end

	   	time = sys.clock() - time

   		print('>>>> Epoch = ' .. epoch .. ', current loss = ' .. cur_loss .. ', time cost = ' .. (time*1000) .. 'ms')
   		-- val(epoch)
   		-- write the loss since this epoch to the log
   		logger:add{epoch, cur_loss, time, acc, mean_acc, mean_iu, fw_iu}
   		-- logger:style{'+-','+-','+-','+-','+-','+-'}   		
		trainset:shuffle()
		-- save the preliminary model
		if epoch%10 == 0 then
			torch.save('fcn8_pre' .. epoch .. '.t7', fcn_net)
		end
	end
	-- logger:plot()
end

-- run
train()
