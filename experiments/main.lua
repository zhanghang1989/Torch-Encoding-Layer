--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- modified from https://github.com/facebook/fb.resnet.torch
-- original copyrights preserves
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'encoding'
csv = require 'csvigo'

package.path = package.path .. ';./?.lua'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)

torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local function istable(x)
   return type(x) == 'table' and not torch.typename(x)
end

print('Total Epochs is ', opt.nEpochs)

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(epoch, valLoader)
	 if istable(trainTop1) then
   	csvf1 = csv.File(paths.concat(opt.save, 'ErrTracking1.csv'), 'a')
	 	csvf1:write({epoch, trainTop1[1], trainTop5[1], trainLoss, testTop1[1], testTop5[1]})
   	csvf1:close()
   	csvf2 = csv.File(paths.concat(opt.save, 'ErrTracking2.csv'), 'a')
	 	csvf2:write({epoch, trainTop1[2], trainTop5[2], trainLoss, testTop1[2], testTop5[2]})
   	csvf2:close()
	 else
   	csvf = csv.File(paths.concat(opt.save, 'ErrTracking.csv'), 'a')
	 	csvf:write({epoch, trainTop1, trainTop5, trainLoss, testTop1, testTop5})
   	csvf:close()
	 end
   local bestModel = false
   if istable(testTop1) then
	    if testTop1[2] < bestTop1 then
      	bestModel = true
      	bestTop1 = testTop1[2]
      	bestTop5 = testTop5[2]
      	print(' * Best model for set 2', bestTop1, bestTop5)
			end
	 elseif testTop1< bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', bestTop1, bestTop5)
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', 
			bestTop1, bestTop5))
