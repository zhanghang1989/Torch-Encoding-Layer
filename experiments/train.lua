--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- modified from https://github.com/facebook/fb.resnet.torch
-- original copyrights preserves
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

local optim = require 'optim'
require 'encoding'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

local function istable(x)
   return type(x) == 'table' and not torch.typename(x)
end

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
	if opt.ft and opt.lockEpoch > 0 then
		print('Locking the Features for Fine-tuning')
		-- only work for FT with encoding
		self.lockEpoch = opt.lockEpoch
		print(model:get(1):get(2))
   	self.params, self.gradParams = model:get(1):get(2):getParameters()
   	self.allparams, self.allgradParams = model:getParameters()
	else
		self.lockEpoch = -1
   	self.params, self.gradParams = model:getParameters()
	end
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

	-- release the lock
	if epoch == self.lockEpoch+1 then
		print('Unlocking the Features for Fine-tuning')
   	self.params, self.gradParams = self.params, self.gradParams
	end

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run(epoch) do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
			local output = self.model:forward(self.input)
      local batchSize
			if istable(output) then
      		output = {
						output[1]:float(),
						output[2]:float(),
					}
				batchSize = output[1]:size(1)
			else
      	output = output:float()
				batchSize = output:size(1)
			end

      local loss = self.criterion:forward(self.model.output, self.target)
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)
      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1)
			if istable(top1) then
				if istable(top1Sum) then

				else
					top1Sum = {0.0, 0.0}
					top5Sum = {0.0, 0.0}
				end
      	top1Sum[1] = top1Sum[1] + top1[1]*batchSize
      	top5Sum[1] = top5Sum[1] + top5[1]*batchSize
      	top1Sum[2] = top1Sum[2] + top1[2]*batchSize
      	top5Sum[2] = top5Sum[2] + top5[2]*batchSize
      	lossSum = lossSum + loss*batchSize
      	print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  set1-top1 %7.3f  set2-top1 %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1[1], top1[2]))
			else
      	top1Sum = top1Sum + top1*batchSize
      	top5Sum = top5Sum + top5*batchSize
      	lossSum = lossSum + loss*batchSize
      	print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))
			end
      N = N + batchSize


      -- check that the storage didn't get changed do to an unfortunate getParameters call
      -- assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
			collectgarbage()
   end

		if istable(top1Sum) then
      	top1Sum[1] = top1Sum[1] / N
      	top5Sum[1] = top5Sum[1] / N
      	top1Sum[2] = top1Sum[2] / N
      	top5Sum[2] = top5Sum[2] / N
   		return top1Sum , top5Sum , lossSum / N
		else
   		return top1Sum / N, top5Sum / N, lossSum / N
		end
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input)
			local batchSize
			if istable(output) then
      		output = {
						output[1]:float(),
						output[2]:float(),
					}
				batchSize = output[1]:size(1) / nCrops
			else
      	output = output:float()
				batchSize = output:size(1) / nCrops
			end

      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
			if istable(top1) then
				if istable(top1Sum) then

				else
					top1Sum = {0.0, 0.0}
					top5Sum = {0.0, 0.0}
				end
      	top1Sum[1] = top1Sum[1] + top1[1]*batchSize
      	top5Sum[1] = top5Sum[1] + top5[1]*batchSize
      	top1Sum[2] = top1Sum[2] + top1[2]*batchSize
      	top5Sum[2] = top5Sum[2] + top5[2]*batchSize
      	print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  set1-top1 %7.3f (%7.3f)  set2-top1 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1[1], top1Sum[1] / N, top1[2], top1Sum[2] / N))
			else
      	top1Sum = top1Sum + top1*batchSize
      	top5Sum = top5Sum + top5*batchSize
      	print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))
			end
      N = N + batchSize


      timer:reset()
      dataTimer:reset()
			collectgarbage()
   end
   self.model:training()
	
		if istable(top1Sum) then
   		print((' * Finished epoch # %d    set1-top1: %7.3f  set2-top1: %7.3f\n'):format(
      epoch, top1Sum[1] / N, top1Sum[2] / N))
      	top1Sum[1] = top1Sum[1] / N
      	top5Sum[1] = top5Sum[1] / N
      	top1Sum[2] = top1Sum[2] / N
      	top5Sum[2] = top5Sum[2] / N
   		return top1Sum , top5Sum 
		else
   		print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))
   		return top1Sum / N, top5Sum / N
		end
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize 
	 local predictions, correct 
	 if istable(output) then
				batchSize = output[1]:size(1)
				predictions = {}
				correct = {}
				_ , predictions[1] = output[1]:float():sort(2, true) -- descending
				_ , predictions[2] = output[2]:float():sort(2, true) -- descending

   			-- Find which predictions match the target
   			correct[1] = predictions[1]:eq(
      	target[1]:long():view(batchSize, 1):expandAs(output[1]))
   			correct[2] = predictions[2]:eq(
      	target[2]:long():view(batchSize, 1):expandAs(output[2]))

				-- Top-1 score
   			local top1 = {1.0 - (correct[1]:narrow(2, 1, 1):sum() / batchSize),
						1.0 - (correct[2]:narrow(2, 1, 1):sum() / batchSize)}
   			-- Top-5 score, if there are at least 5 classes
   			local len1 = math.min(5, correct[1]:size(2))
   			local len2 = math.min(5, correct[2]:size(2))
   			local top5 = {1.0 - (correct[1]:narrow(2, 1, len1):sum() / batchSize),
   									1.0 - (correct[2]:narrow(2, 1, len2):sum() / batchSize)}


   			return {top1[1] * 100, top1[2] * 100}, {top5[1] * 100, top5[2] * 100}
	 else
				batchSize = output:size(1)
				_ , predictions = output:float():sort(2, true) -- descending

   			-- Find which predictions match the target
   			correct = predictions:eq(
      	target:long():view(batchSize, 1):expandAs(output))
   			
				-- Top-1 score
   			local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   			-- Top-5 score, if there are at least 5 classes
   			local len = math.min(5, correct:size(2))
   			local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   			return top1 * 100, top5 * 100
	 end
   

end

function Trainer:copyInputs(sample)
	-- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
	-- if using DataParallelTable. The target is always copied to a CUDA tensor
	if istable(sample.input) then
		self.input = self.input or (self.opt.nGPU == 1
			and {torch.CudaTensor(), torch.CudaTensor()} or 
			{cutorch.createCudaHostTensor(), cutorch.createCudaHostTensor()})
		self.target = self.target or {torch.CudaTensor(), torch.CudaTensor()}
		self.input[1]:resize(sample.input[1]:size()):copy(sample.input[1])
		self.input[2]:resize(sample.input[2]:size()):copy(sample.input[2])
		self.target[1]:resize(sample.target[1]:size()):copy(sample.target[1])
		self.target[2]:resize(sample.target[2]:size()):copy(sample.target[2])
	else
   	self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
  	self.target = self.target or torch.CudaTensor() 
  	self.input:resize(sample.input:size()):copy(sample.input)
   	self.target:resize(sample.target:size()):copy(sample.target)
	end
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'stl10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'joint' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.ft then
      decay = epoch > 40 and 2 or 1
   else
      decay = epoch >= 40 and 2 or 1
   end
	 local learningRate = self.opt.LR * math.pow(0.1, decay)
	 print('Learning Rate is ', learningRate)
   return learningRate 
end

return M.Trainer
