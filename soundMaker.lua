require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'getvocab'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'
require 'window'
require 'yHat'
local mixture = require 'mixtureGauss'
local model_utils=require 'model_utils'
require 'cunn'
require 'distributions'
local matio = require 'matio'

function getX(output)
    -- local piStart = 1
    -- local piEnd = opt.numMixture
    -- local pi_t = output[{{},{piStart,piEnd}}]

    -- local sizeMeanInput = opt.inputSize * opt.numMixture
    -- local sizeCovarianceInput = opt.inputSize * opt.numMixture

    -- local muStart = piEnd + 1
    -- local muEnd = piEnd + sizeMeanInput

    -- local sigmaStart = muEnd + 1
    -- local sigmaEnd = muEnd + sizeCovarianceInput	

    pi_t, mu_t, u_t = unpack(output)     

    local chosenPi = torch.multinomial(torch.exp(pi_t), 1):squeeze()

    local chosenMuStart = ((chosenPi - 1) * opt.inputSize)
    local chosenMuEnd = chosenMuStart + (opt.inputSize - 1)

    local chosenSigmaStart = ((chosenPi - 1) * opt.inputSize * opt.dimSize)
    local chosenSigmaEnd = chosenSigmaStart + ((opt.inputSize * opt.dimSize) - 1)

    local mu_t = mu_t[{{},{chosenMuStart,chosenMuEnd}}]:squeeze()

    local u_t = u_t[{{},{chosenSigmaStart,chosenSigmaEnd}}]:squeeze()

    local sigma = u_t:t()*u_t

    sample = distributions.mvn.rnd(mu_t:double(), sigma:double())

    return sample:float():resize(1,opt.inputSize)
end

local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for training model.')

cmd:option('-inputSize' , 35, 'number of input dimension')
cmd:option('-hiddenSize' , 600, 'number of hidden units in lstms')
cmd:option('-dimSize' , 2, 'number of hidden units in lstms')
cmd:option('-maxlen' , 500, 'max sequence length')
cmd:option('-isCovarianceFull' , false, 'true if full covariance, o.w. diagonal covariance')
cmd:option('-numMixture' , 20, 'number of mixture components in output layer') 
cmd:option('-modelFilename' , 'model_exlayer.t7', 'model filename') 
cmd:option('-testString' , 'but i do think they can have', 'string for testing') 

cmd:text()
opt = cmd:parse(arg)

cuMat = getOneHotStrs({[1]=opt.testString})

model = torch.load(opt.modelFilename)
param, grad = model.rnn_core:getParameters()
--param:uniform(-0.008,0.008)

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(1, opt.hiddenSize):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()
initstate_h3_c = initstate_h1_c:clone()
initstate_h3_h = initstate_h1_c:clone()
initstate_h4_c = initstate_h1_c:clone()
initstate_h4_h = initstate_h1_c:clone()

-- initialize input
x = torch.zeros(1,opt.inputSize)

function getInitW(cuMat)
	cuMatClone = cuMat:clone()
    return torch.zero(cuMatClone[{{},{1},{}}]:squeeze(2))
end

-- initialize window to first char in all elements of the batch
local w = {[0]=getInitW(cuMat:cuda())}

local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
local lstm_c_h4 = {[0]=initstate_h3_c} -- internal cell states of LSTM
local lstm_h_h4 = {[0]=initstate_h3_h} -- output values of LSTM

local kappa_prev = {[0]=torch.zeros(1,10):cuda()}

local output_h1_w = {}
local input_h3_y = {}
local output_h3_y = {}

-- FORWARD

for t = 1, opt.maxlen - 1 do
    -- model 
    output_y, kappa_prev[t], w[t], phi, lstm_c_h1[t], lstm_h_h1[t],
    lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t], lstm_c_h4[t], lstm_h_h4[t]
	= unpack(model.rnn_core:forward({x:cuda(), cuMat:cuda(), 
         kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
         lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1], lstm_c_h4[t-1], lstm_h_h4[t-1]}))

	-- perform op on x
	x = getX(output_y)
	if straightMat then
		straightMat = torch.cat(straightMat, x, 1)
	else
		straightMat = x
	end
end

--torch.save('STRGHT.t7', straightMat)
--matio.save('STRGHT.mat',straightMat)

straightMat:mul(0.80)
std = torch.load('toy_std_35.t7')
mean1 = torch.load('toy_mean_35.t7')
rs_std = torch.expand(std, 499, 35)
rs_mean = torch.expand(mean1, 499, 35)
newin = torch.cmul(straightMat:float(), rs_std:float())
newin = newin:float() + rs_mean:float()
matio.save('strght1.mat', newin)

