require 'image'
train_indices = {}
data_dir = "../"

local train_f = io.open(data_dir.."VOC2011/ImageSets/Segmentation/trainval.txt")
if train_f then
    for line in train_f:lines() do
        table.insert(train_indices, line)
    end
else
end

print(#train_indices)
i = 1
sample = {}
label_file = data_dir..'VOC2011/SegmentationClass/'..train_indices[i]..'.png'
label = image.load(label_file, 3, 'byte')
print(#label)
print(train_indices[i])

