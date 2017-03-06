function fast_hist(a,b,n)
	-- a Byte tensor indicating whether each element of a>=0;if true return 1,if false return 0
	p1 = torch.ge(a,0)
	-- a<n
	q1 = torch.lt(a,n)
	k = torch.cmul(p1,q1)
    return 
    -- np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)
   

function compute_hist(net,dataset,layer = 'score',gt = 'label')
n_cl = 
loss =0
hist = torch.zeros(n_cl,n_cl)

--p = net.blobs[gt].data[0, 0]
--q = net.blobs[layer].data[0].argmax(0)

for idx in dataset do
    local net= fcn_net:forward(batchInputs)
    hist:add(fast_hist(a:view(p:nElement()),q:nElement()),n_cl))                              
 
    --loss:add(net.blobs['loss'].data.flat[0])
    return hist, torch.cdiv(loss,len(dataset))
                            
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
u = torch.sum(hist,1)+torch.sum(hist,2)-torch.diag(hist)
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
-- fiu = torch.cdiv(fw,u)
-- ?   print '>>>','Iteration', iter, 'fwavacc', \(freq[freq > 0] * iu[freq > 0]).sum()
