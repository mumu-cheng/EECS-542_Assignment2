
-- the number of class
n_cl = 21
loss = 0
-- hist is a n_cl*n_cl tensor; hist[i][j] is the number of pixels of class i predicted to belong to class j 
hist ＝ torch.zeros(n_cl，n_cl)

-- res is the result label image we get (HxW)
-- expected is the correct label image (HxW)
function compute_hist(result, groundtruth, hist)
for i = 1,result:size(1)
   for j =1,result:size(2)
     hist[groundtruth[i][j]][result[i][j]]:add(1)
   end
end

-- the total number of pixels
t = torch.sum(hist)
-- the total number of pixels of class i
ti = torch.sum(hist,2)
-- the total number of pixels which have been correctly classified
m = torch.trace(hist)
-- the number of correctly classified pixel for each class
mi = torch.diag(hist)
-- intersection 
u = torch.add(torch.sum(hist,1),ti)
u:csub(torch.diag(hist))
-- frequency weighted
fw = torch.div(toch.sum(torch.cmul(ti,mi)),t)


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

--frequency weighted IU ?   
freq = torch.div(ti,t)
fk1 = torch.cmul(freq,torch.gt(freq,0))
fk2 = torch.cmul(iu,torch.gt(freq,0))
fk = torch.sum(torch.cmul(fk1,fk2))
fiu = torch.cdiv(fw,fk)
print('>>>','Iteration', iter, 'fwavacc', fiu)
