--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Created by: Hang Zhang
-- ECE Department, Rutgers University
-- Email: zhang.hang@rutgers.edu
-- Copyright (c) 2016
--
-- Feel free to reuse and distribute this software for research or 
-- non-profit purpose, subject to the following conditions:
--  1. The code must retain the above copyright notice, this list of
--     conditions.
--  2. Original authors' names are not deleted.
--  3. The authors' names are not used to endorse or promote products
--      derived from this software 
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

local M = {}

local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end


function M.exec(opt, cacheFile)
	local dir = paths.dirname(cacheFile)
	local cifarPath = string.format('%s/cifar10.t7', dir)
	local stlPath = string.format('%s/stl10.t7', dir)
	if not paths.filep(cifarPath) or not isvalid(opt, cifarPath) then
		paths.mkdir('gen')
		local script = paths.dofile('cifar10-gen.lua')
		script.exec(opt, cifarPath)
	end
	if not paths.filep(stlPath) or not isvalid(opt, stlPath) then
		paths.mkdir('gen')
		local script = paths.dofile('stl10-gen.lua')
		script.exec(opt, stlPath)
	end
	local cifarData = torch.load(cifarPath)
	local stlData = torch.load(stlPath)
   
   print(" | saving joint dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = { 
				set1 = cifarData.train,
				set2 = stlData.train,
			},
      val = {
				set1 = cifarData.val,
				set2 = stlData.val,
			},
   })
end

return M
