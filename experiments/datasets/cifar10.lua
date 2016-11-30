--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- modified from https://github.com/facebook/fb.resnet.torch
-- original copyrights preserves
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

local t = require 'datasets/transforms'

local M = {}
local CifarDataset = torch.class('resnet.CifarDataset', M)

function CifarDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function CifarDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function CifarDataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire CIFAR-10 training set
local meanstd = {
   mean = {125.3, 123.0, 113.9},
   std  = {63.0,  62.1,  66.7},
}

function CifarDataset:preprocess(opt)
   if self.split == 'train' then
			if opt.multisize then
      	return t.Compose{
       		t.ColorNormalize(meanstd),
         	t.HorizontalFlip(0.5),
         	--t.RandomTwoCrop(24, 36, 4),
         	t.RandomThreeCrop(28, 32, 36, 4),
      	}
			else
      	return t.Compose{
       		t.ColorNormalize(meanstd),
         	t.HorizontalFlip(0.5),
         	t.RandomCrop(32, 4),
      	}
			end
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CifarDataset
