require 'nn'
require 'nngraph'
require 'paths'
require 'cunn'
require 'cudnn'
require 'optim'
require 'cutorch'
require 'image'

print('>>>> Start loading validation dataset')

-- load the validation set
valset = torch.load('../Datasett7/valset.t7')

-- load the val indices of all images
val_indices = {}
data_dir = '../'
local val_f = io.open(data_dir.."VOC2011/ImageSets/Segmentation/val.txt")
if val_f then
    for line in val_f:lines() do
        table.insert(val_indices, line)
    end
else
end


-- load the functions to calculate metrics 
paths.dofile('metrics.lua')

paths.dofile('getLabelMap.lua')
PixelMap = {}

for k, v in pairs(labelMap) do
	PixelMap[v] = k
end
-- load the untrained model
-- paths.dofile('fcn8.lua')
paths.dofile('CropTable.lua')
fcn_net = torch.load('fcn8_2_125.t7')
fcn_net = fcn_net:cuda()
print('>>>> Finish loading net and converting net to cuda')

-- softmax layer for getting the final result
softmax_layer = nn.SpatialSoftMax():cuda()

for index = 1, valset:size() do
	val_image = valset[index][1]:cuda()
	true_seg = valset[index][2]:cuda()
	net_seg = fcn_net:forward(val_image)
	net_seg = softmax_layer:forward(net_seg)
	_, net_seg = torch.max(net_seg,2)
	net_seg = net_seg:squeeze()
	-- print(#net_seg)
	local H = net_seg:size()[1]
	local W = net_seg:size()[2]
	local seg_img = torch.zeros(3, H, W)

	for i = 1, H do
		for j = 1, W do
			local pixel = PixelMap[net_seg[i][j]]
			local channel3 = pixel % 255
			pixel = torch.floor(pixel / 255)
			local channel2 = pixel % 255
			channel1 = torch.floor(pixel / 255)
			seg_img[1][i][j] = channel1
			seg_img[2][i][j] = channel2
			seg_img[3][i][j] = channel3
		end
	end

	image.save("../SegmentRes/"..val_indices[index]..".png", seg_img/255)
	hist = compute_hist(net_seg,true_seg:squeeze()) 
end

calculate_metrics(hist)







