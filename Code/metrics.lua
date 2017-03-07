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
	   	if gtr[i][j] == 255 then
	   	else 
	   		hist[gtr[i][j]][res[i][j]] = hist[gtr[i][j]][res[i][j]] + 1
	   	end
	   end
	end
	return hist
end

-- calculate metrics
function calculate_metrics()
	-- some variable to calculate final metrics
	-- the total number of pixels
	local t = torch.sum(hist)
	-- the total number of pixels of class i
	local ti = torch.sum(hist,2)
	-- the total number of pixels which have been correctly classified
	local n = torch.trace(hist)
	-- the number of correctly classified pixel for each class
	local ni = torch.diag(hist)
	-- union 
	local u = torch.add(torch.sum(hist,1),ti)
	u:csub(torch.diag(hist))
	-- frequency weighted
	local fw = torch.div(toch.sum(torch.cmul(ti,ni)),t)
	-- pixel accuracy
	acc = torch.div(n,t)
	print('>>> ' .. 'epoch ' .. epoch .. ' -- pixel accuracy ' .. acc)
	-- mean accuracy / per-class accuracy
	per_acc = torch.cdiv(ni,ti)
	mean_acc = torch.sum(per_acc) / n_cl
	print('>>> ' .. 'epoch ' .. epoch .. ' -- mean accuracy ' .. mean_acc)
	-- mean IU / per-class IU
	iu = torch.cdiv(ni,u)
	mean_iu = torch.sum(iu) / n_cl
	print('>>> ' .. 'epoch ' .. epoch .. ' -- mean IU ' .. mean_iu)
	-- frequency weighted IU
	num = torch.cmul(ti,ni)
	det = torch.cdiv(num,u)
	fwiu = torch.sum(det) / t
	print('>>> ' .. 'epoch ' .. epoch .. ' -- frequency weighted IU' .. fwiu)
end


