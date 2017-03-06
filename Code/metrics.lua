-- functions to calculate all four metrics

local n_cl = 21 -- the number of class

-- hist is a n_cl*n_cl tensor
-- hist[i][j] is the number of pixels of class i predicted to belong to class j 
local hist = torch.zeros(n_cl,n_cl)

-- res is the result label image we get (HxW)
-- gtr is the ground_truth label image (HxW)
function compute_hist(res, gtr)
	for i = 1,res:size(1) do
	   for j = 1,res:size(2) do
	   	hist[gtr[i][j]][res[i][j]] = hist[gtr[i][j]][res[i][j]] + 1
	   end
	end
	return hist
end

-- calculate statistical values 
function prepare_metrics()
	-- some variable to calculate final metrics
	-- the total number of pixels
	local t = torch.sum(hist)
	-- the total number of pixels of class i
	local ti = torch.sum(hist,2)
	-- the total number of pixels which have been correctly classified
	local m = torch.trace(hist)
	-- the number of correctly classified pixel for each class
	local mi = torch.diag(hist)
	-- union 
	local u = torch.add(torch.sum(hist,1),ti)
	u:csub(torch.diag(hist))
	-- frequency weighted
	local fw = torch.div(toch.sum(torch.cmul(ti,mi)),t)
	return
end


-- pixel accuracy
local function cal_pixel_accuracy()

end
-- mean loss
print('>>>','Iteration',iter,'loss',loss)
-- pixel accuracy
acc = torch.div(m,t)
print('>>>','Iteration',iter,'overall accuracy ',acc)
-- per-class accuracy
per_acc = torch.cdiv(mi,ti)
print('>>>','Iteration',iter,'per class accuracy ',per_acc)
-- per-class IU
iu = torch.cdiv(mi,u)
print('>>>','Iteration',iter,'per-class IU ',iu)
print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
--frequency weighted IU ?   
freq = torch.div(ti,t)
fk1 = torch.cmul(freq,torch.gt(freq,0))
fk2 = torch.cmul(iu,torch.gt(freq,0))
fk = torch.sum(torch.cmul(fk1,fk2))
fiu = torch.cdiv(fw,fk)
print('>>>','Iteration', iter, 'fwavacc', fiu)



-- pixel accuracy 
local function cal_mean_accuracy()
end
-- mean IU
local function cal_iu()
end
-- frequency weighted IU
local function cal_fw_iu()
end