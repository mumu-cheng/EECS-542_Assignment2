require 'loadcaffe'
require 'paths'

pretrained = loadcaffe.load('../../deploy.prototxt', '../../fcn8s-heavy-pascal.caffemodel', 'nn')
paths.dofile('fcn32.lua')

param_1 = fcn_net:parameters()
param_2 = pretrained:parameters()

-- conv1
param_1[1]:copy(param_2[1])
param_1[2]:copy(param_2[2])
param_1[3]:copy(param_2[3])
param_1[4]:copy(param_2[4])
-- conv2
param_1[5]:copy(param_2[5])
param_1[6]:copy(param_2[6])
param_1[7]:copy(param_2[7])
param_1[8]:copy(param_2[8])
-- conv3
param_1[9]:copy(param_2[9])
param_1[10]:copy(param_2[10])
param_1[11]:copy(param_2[11])
param_1[12]:copy(param_2[12])
param_1[13]:copy(param_2[13])
param_1[14]:copy(param_2[14])
-- conv4
param_1[15]:copy(param_2[15])
param_1[16]:copy(param_2[16])
param_1[17]:copy(param_2[17])
param_1[18]:copy(param_2[18])
param_1[19]:copy(param_2[19])
param_1[20]:copy(param_2[20])
-- conv5
param_1[21]:copy(param_2[21])
param_1[22]:copy(param_2[22])
param_1[23]:copy(param_2[23])
param_1[24]:copy(param_2[24])
param_1[25]:copy(param_2[25])
param_1[26]:copy(param_2[26])
-- fcn
param_1[27]:copy(param_2[27])
param_1[28]:copy(param_2[28])
param_1[29]:copy(param_2[29])
param_1[30]:copy(param_2[30])
-- score
param_1[31]:copy(param_2[31])
param_1[32]:copy(param_2[32])
param_1[33] = param_1[33]/10000

torch.save('pre_fcn32.t7',fcn_net)