--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Created by: Hang Zhang
-- ECE Department, Rutgers University
-- Email: zhang.hang@rutgers.edu
-- Copyright (c) 2016
--
-- Free to reuse and distribute this software for research or 
-- non-profit purpose, subject to the following conditions:
--  1. The code must retain the above copyright notice, this list of
--     conditions.
--  2. Original authors' names are not deleted.
--  3. The authors' names are not used to endorse or promote products
--      derived from this software 
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


local nn = require 'nn'
require 'cunn'
require 'cudnn'
require 'encoding'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization


local function createModel(opt)
	local depth = opt.depth
	local shortcutType = opt.shortcutType or 'B'
	local iChannels

	-- The shortcut layer is either identity or 1x1 convolution
	local function shortcut(nInputPlane, nOutputPlane, stride)
		local useConv = shortcutType == 'C' or
			(shortcutType == 'B' and nInputPlane ~= nOutputPlane)
	if useConv then
		-- 1x1 convolution
		return nn.Sequential()
			:add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
	elseif nInputPlane ~= nOutputPlane then
		-- Strided, zero-padded identity shortcut
		return nn.Sequential()
			:add(nn.SpatialAveragePooling(1, 1, stride, stride))
			:add(nn.Concat(2)
				:add(nn.Identity())
				:add(nn.MulConstant(0)))
		else
			return nn.Identity()
		end
	end
   
	local function ShareGradInput(module, key)
		assert(key)
		module.__shareGradInputKey = key
		return module
	end

	local function basicblock(n, stride, type)
		local nInputPlane = iChannels
		iChannels = n

		local block = nn.Sequential()
		local s = nn.Sequential()
		if type == 'both_preact' then
			block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
			block:add(ReLU(true))
		elseif type ~= 'no_preact' then
			s:add(SBatchNorm(nInputPlane))
			s:add(ReLU(true))
		end
		s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
		s:add(SBatchNorm(n))
		s:add(ReLU(true))
		s:add(Convolution(n,n,3,3,1,1,1,1))
		
		return block
			:add(nn.ConcatTable()
				:add(s)
				:add(shortcut(nInputPlane, n, stride)))
				:add(nn.CAddTable(true))
   end

	local function bottleneck(n, stride, type)
		local nInputPlane = iChannels
		iChannels = n * 4

		local block = nn.Sequential()
		local s = nn.Sequential()
		if type == 'both_preact' then
			block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
			block:add(ReLU(true))
		elseif type ~= 'no_preact' then
			s:add(SBatchNorm(nInputPlane))
			s:add(ReLU(true))
      end
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))

			return block
				:add(nn.ConcatTable()
					:add(s)
					:add(shortcut(nInputPlane, n * 4, stride)))
				:add(nn.CAddTable(true))
	end
	-- Creates count residual blocks with specified number of features
	local function layer(block, features, count, stride, type)
		local s = nn.Sequential()
		if count < 1 then
			return s
		end
		s:add(block(features, stride,
				type == 'first' and 'no_preact' or 'both_preact'))
		for i=2,count do
			s:add(block(features, 1))
		end
		return s
	end
	
	local model = nn.Sequential()
	if opt.dataset == 'cifar10' or opt.dataset == 'stl10'  then
		print('opt.bottleneck', opt.bottleneck)
		if opt.bottleneck then

		else
			-- Model type specifies number of layers for CIFAR-10 model
			assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56,.. 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | Encoding-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(layer(basicblock, 16, n))
      model:add(layer(basicblock, 32, n, 2))
			model:add(layer(basicblock, 64, n, 2))
      model:add(nn.View(64, -1):setNumInputDims(3))
			model:add(nn.Transpose({2,3}))
			model:add(nn.Encoding(opt.nCodes, 64))
      model:add(nn.View(-1):setNumInputDims(2))
			model:add(nn.Normalize(2))
      model:add(nn.Linear(64*opt.nCodes, 10))
			print(model)
		end
	elseif opt.dataset == 'joint' then
		assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56,.. 1202')
    local n = (depth - 2) / 6
    iChannels = 16
		-- joint encoding for cifar10 and stl10
		local m1 = nn.Sequential()
		m1:add(Convolution(3,16,3,3,1,1,1,1))
    m1:add(SBatchNorm(16))
    m1:add(ReLU(true))
		m1:add(layer(basicblock, 16, n))
    m1:add(layer(basicblock, 32, n, 2))
    m1:add(layer(basicblock, 64, n, 2))
   	m1:add(nn.View(64,-1):setNumInputDims(3))
		m1:add(nn.Transpose({2,3}))
		-- sharing weights for joint training
		local m2 = m1:clone('weight','bias','gradWeight','gradBias');

		local model1=nn.Sequential()
		model1:add(m1)
		model1:add(nn.Encoding(opt.nCodes, 64))
		model1:add(nn.View(-1):setNumInputDims(2))
		model1:add(nn.Normalize(2))
		model1:add(nn.Linear(64*opt.nCodes, 10))

		local model2=nn.Sequential()
		model2:add(m2)
		model2:add(nn.Encoding(opt.nCodes, 64))
		model2:add(nn.View(-1):setNumInputDims(2))
		model2:add(nn.Normalize(2))
		model2:add(nn.Linear(64*opt.nCodes, 10))
		
		model = nn.ParallelTable()
			:add(model1)
			:add(model2)
			
		print(model)
	else
      error('invalid dataset: ' .. opt.dataset)
	end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
