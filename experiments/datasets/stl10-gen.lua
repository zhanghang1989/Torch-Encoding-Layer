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

local URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

local M = {}

local function convertToTensor(inputFname, inputLabelsFname)
    local nSamples = 0
    local m=torch.DiskFile(inputFname, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamplesF = length / (3*96*96) 
    assert(nSamplesF == math.floor(nSamplesF), 'expecting numSamples to be an exact integer')
    nSamples = nSamples + nSamplesF
    m:close()

    local data = torch.ByteTensor(nSamples, 3, 96, 96)
    local index = 1
    local m=torch.DiskFile(inputFname, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamplesF = length / (3*96*96)
    m:seek(1)
    for j=1,nSamplesF do
        local store = m:readByte(3*96*96)
        data[index]:copy(torch.ByteTensor(store))
        index = index + 1
    end
    m:close()

	local m=torch.DiskFile(inputLabelsFname, 'r'):binary()
  local labels = torch.ByteTensor(m:readByte(nSamplesF)),
  m:close()
	return {
      data = data:transpose(3,4),--:view(-1, 3, 96, 96),
			labels=labels,
	}
end

function M.exec(opt, cacheFile)
   print("=> Downloading STL-10 dataset from " .. URL)
   local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
   assert(ok == true or ok == 0, 'error downloading STL-10')


	local trainData = convertToTensor('gen/stl10_binary/train_X.bin',
					'gen/stl10_binary/train_y.bin')
	local testData = convertToTensor('gen/stl10_binary/test_X.bin',
					'gen/stl10_binary/test_y.bin')
   
   print(" | saving STL-10 dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M
