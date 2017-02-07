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

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findClasses(dir)
   -- copied from fb.resnet.torch
   local dirs = paths.dir(dir)
   table.sort(dirs)

   local classList = {}
   local classToIdx = {}
   for _ ,class in ipairs(dirs) do
      if not classToIdx[class] and class ~= '.' and class ~= '..' then
         table.insert(classList, class)
         classToIdx[class] = #classList
      end
   end

   return classList, classToIdx
end

local function findImages(dir, classToIdx, append, idx)
   -- copied from fb.resnet.torch
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()
   -- read the txt
	 print('reading the file')
	 print(dir .. append .. string.format('%i.txt', idx))
   local file = io.open(dir .. append .. string.format('%i.txt', idx), 'r')
	 local f = io.input(file)
   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local className = paths.basename(paths.dirname(line))
      local filename = paths.basename(line)
      local path = className .. '/' .. filename

      local classId = classToIdx[className]
      assert(classId, 'class not found: ' .. className)

      table.insert(imagePaths, path)
      table.insert(imageClasses, classId)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   local imageClass = torch.LongTensor(imageClasses)
   return imagePath, imageClass
end

function M.exec(opt, cacheFile)
   -- copied from fb.resnet.torch
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local imgDir   = paths.concat(opt.data, 'images')
   local labelDir = paths.concat(opt.data, 'labels')
   assert(paths.dirp(imgDir), 'image directory not found: ' .. imgDir)

   print("=> Generating list of images")
   local classList, classToIdx = findClasses(imgDir)

	 -- TODO FIXME idx = opts.fold;
	 idx = 1;
   print(" | finding all training images")
   local trainImagePath, trainImageClass = findImages(labelDir, classToIdx, '/train', idx)
   print(" | finding all test images")
   local testImagePath , testImageClass  = findImages(labelDir, classToIdx, '/test' , idx)
   print(" | finding all val images")
   local valImagePath  , valImageClass   = findImages(labelDir, classToIdx, '/validate'  , idx)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
      },
      test = {
         imagePath = testImagePath,
         imageClass = testImageClass,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
