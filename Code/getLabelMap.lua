require 'image'

-- borrow shamelessly from http://stackoverflow.com/questions/15706270/sort-a-table-in-lua
function spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

print("Start getting Label Map......")

train_indices = {}
labelMap = {}
data_dir = "../"

-- for i = 0, 255 do
--     labelMap[i] = 0
-- end

local train_f = io.open(data_dir.."VOC2011/ImageSets/Segmentation/trainval.txt")
if train_f then
    for line in train_f:lines() do
        table.insert(train_indices, line)
    end
else
end

batchSize = 50
for i = 1, batchSize do
    if (i % 5 == 0) then
        print("Progress: "..tostring(math.floor(i * 100/batchSize)).."%")
    end
    sample = {}
    label_file = data_dir..'VOC2011/SegmentationClass/'..train_indices[i]..'.png'
    local label = image.load(label_file, 3, 'byte')
    for i = 1, label:size()[2] do
        for j = 1, label:size()[3] do
            local channel1 = label[1][i][j]
            local channel2 = label[2][i][j]
            local channel3 = label[3][i][j]
            labelMap[channel1 * 255 * 255 + channel2 * 255 + channel3] = 1
        end
    end
end

table.sort(labelMap)
labelMap[224 * 255 * 255 + 224 * 255 + 192] = 255
i = 1
for k, v in spairs(labelMap) do
    if labelMap[k] ~= 255 then
        labelMap[k] = i
        i = i + 1
    end
end

print("Finish getting Label Map")

-- print(labelMap)