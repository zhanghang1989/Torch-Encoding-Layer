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

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local LightDataset = torch.class('resnet.LightDataset', M)

function LightDataset:__init(imageInfo, opt, split)
   -- copied from fb.resnet.torch
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = opt.data
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function LightDataset:get(i)
   -- copied from fb.resnet.torch
   local path = ffi.string(self.imageInfo.imagePath[i]:data())
   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]

   return {
      input = image,
      target = class,
   }
end

function LightDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))
			print('before decompressing', path)
      input = image.decompress(b, 3, 'float')
   end

   return input
end

function LightDataset:size()
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function LightDataset:preprocess(opt)
   -- copied from fb.resnet.torch
   if self.split == 'train' then
			if opt.multisize then
      	return t.Compose{
				 t.Scale(400),
         --t.RandomSizedCrop(352),
				 --t.RandomTwoSizeCrop(352),
				 t.RandomTwoCrop(352, 320),
				 --[[
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
				 --]]
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      	}
			else 
      	return t.Compose{
				 t.Scale(400),
         t.RandomCrop(352),
				 --[[
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
				 --]]
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
				}
				end
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(400),
         t.ColorNormalize(meanstd),
         Crop(352),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.LightDataset
