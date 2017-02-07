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

local t = require 'datasets/transforms'

local M = {}
local JointDataset = torch.class('resnet.JointDataset', M)

function JointDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function JointDataset:get(i)
	local idx1 = (i-1) % self.imageInfo.set1.data:size(1) + 1
	local idx2 = (i-1) % self.imageInfo.set2.data:size(1) + 1

	local image1 = self.imageInfo.set1.data[idx1]:float()
	local label1 = self.imageInfo.set1.labels[idx1]
	local image2 = self.imageInfo.set2.data[idx2]:float()
	local label2 = self.imageInfo.set2.labels[idx2]

   return {
      input = {
				image1,
				image2,
				},
      target = {
				label1,
				label2,
				},
   }
end

function JointDataset:size()
   return math.max(self.imageInfo.set1.data:size(1),
	 					self.imageInfo.set2.data:size(1))
end

-- Same Params as in CIFAR-10 training set
local meanstd = {
   mean = {125.3, 123.0, 113.9},
   std  = {63.0,  62.1,  66.7},
}

function JointDataset:preprocess()
   if self.split == 'train' then
			local f1 = t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      	}
				local f2 = t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(96, 12),
      	}
      return function(input)
				return {
					f1(input[1]),
					f2(input[2]),
				}
			end
   elseif self.split == 'val' then
	 		local f = t.ColorNormalize(meanstd)
      return function(input) 
				return{
					f(input[1]),
					f(input[2]),
				}
			end
   else
      error('invalid split: ' .. self.split)
   end
end

return M.JointDataset
