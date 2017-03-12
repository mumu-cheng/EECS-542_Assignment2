require 'nn'
require 'nngraph'
require 'paths'
require 'optim'
require 'image'

print('>>>> Start loading validation dataset')
paths.dofile('getLabelMap.lua')
-- load the validation set
-- valset = torch.load('../Datasett7/valset.t7')

-- load the val indices of all images
function convert_label(label)
    local height = label:size()[2]
    local width = label:size()[3]
    new_label = torch.zeros(1, height, width)
    for i = 1, height do
        for j = 1, width do
            local channel1 = label[1][i][j]
            local channel2 = label[2][i][j]
            local channel3 = label[3][i][j]
            new_label[1][i][j] = labelMap[channel1 * 255 * 255 + channel2 * 255 + channel3]
        end
    end
    -- print(new_label)
    return new_label
end

val_indices = {}
data_dir = '../'
local val_f = io.open(data_dir.."VOC2011/ImageSets/Segmentation/val.txt")
if val_f then
    for line in val_f:lines() do
        table.insert(val_indices, line)
    end
else
end

PixelMap = {}

for k, v in pairs(labelMap) do
	PixelMap[v] = k
end
-- load the untrained model
-- paths.dofile('fcn8.lua')
print(PixelMap)

for index = 1, #val_indices do
    val_label_file = data_dir..'VOC2011/SegmentationClass/'..val_indices[index]..'.png'
    img = image.load(val_label_file, 3, 'byte')
    -- print(img)
    local net_seg = convert_label(img)
    -- print(net_seg)
	local H = net_seg:size()[2]
	local W = net_seg:size()[3]
	local seg_img = torch.zeros(3, H, W)

	for i = 1, H do
		for j = 1, W do
            -- print(net_seg)
            -- print(net_seg[][i][j])
			local pixel = PixelMap[net_seg[1][i][j]]
			local channel3 = pixel % 255
			pixel = torch.floor(pixel / 255)
			local channel2 = pixel % 255
			channel1 = torch.floor(pixel / 255)
			seg_img[1][i][j] = channel1
			seg_img[2][i][j] = channel2
			seg_img[3][i][j] = channel3
            if (img[1][i][j] ~= channel1 or img[2][i][j] ~= channel2 or img[3][i][j] ~= channel3) then
                print("===========NOT EQUAL==========")
                print(i)
                print(j)
                print(img[1][i][j]..img[2][i][j]..img[3][i][j])
                print(net_seg[1][i][j])
                print(channel1..channel2..channel3)
            end
		end
	end
    print("success!")
    -- for i = 1, H do
	-- 	for j = 1, W do
    --         print("========================================")
    --         print(img[1][i][j]..img[2][i][j]..img[3][i][j])
    --         print(seg_img[1][i][j]..seg_img[2][i][j]..seg_img[3][i][j])
    --     end
    -- end
	image.save("../SegmentRes2/"..val_indices[index]..".png", seg_img/255)

end

