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

local function findImages(dir, classToIdx)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local className = paths.basename(paths.dirname(line))
      local filename = paths.basename(line)
      local path = dir .. '/' .. className .. '/' .. filename

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

   local imgDir   = paths.concat(opt.data, 'image')
   assert(paths.dirp(imgDir), 'image directory not found: ' .. imgDir)

   print("=> Generating list of images")
   local classList, classToIdx = findClasses(imgDir)

	 idx = 1;
   print(" | finding all training images")
   local ImagePath, ImageClass = findImages(imgDir, classToIdx)
	
	 -- Follow the standard in prior work (https://github.com/mcimpoi/deep-fbanks/blob/master/os_train.m#L132).
	 trainImagePath = torch.cat(ImagePath:view(10,100,-1):narrow(2,1,40), ImagePath:view(10,100,-1):narrow(2,51,50), 2):view(900,-1)
	 valImagePath 	= ImagePath:view(10,100,-1):narrow(2,41,10):reshape(100,83)
	 trainImageClass = torch.cat(ImageClass:view(10,100):narrow(2,1,40), ImageClass:view(10,100):narrow(2,51,50), 2):view(900)
	 valImageClass	 = ImageClass:view(10,100):narrow(2,41,10):reshape(100)

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
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
