--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- modified from https://github.com/facebook/fb.resnet.torch
-- original copyrights preserves
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

local M={}

function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	-- Data
	cmd:option('-dataset',    'joint','Options: ')
	cmd:option('-multisize',  'false',       'Path to dataset')
	cmd:option('-threesize',  'false',       'Path to dataset')
	cmd:option('-data',       '',       'Path to dataset')
	cmd:option('-nSplit',     1,        'Current number of split')
	cmd:option('-nThreads',   8,        'Threads for data loading')
	cmd:option('-gen',        'gen',      'Path to save generated files')
	-- Model
	cmd:option('-netType',    'encoding', 'Options: resnet | preresnet | encoding')
	cmd:option('-nCodes',     16,       'Options: 2 ~ inf')
	cmd:option('-depth',      20,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
	cmd:option('-bottleneck', 'false',  'Options: true | false')
	-- Checkpointing 
	cmd:option('-save',       'untitle','Directory in which to save')
	cmd:option('-resume',     'none',   'Resume in this directory')

	-- Training
	cmd:option('-nGPU',       1,        'Number of GPUs, 1 by default')
	cmd:option('-batchSize',  128,      'Batch size, 128 by default')
	cmd:option('-nEpochs',         0,       'Number of total epochs to run')
	cmd:option('-shareGradInput','false','Share gradInput to reduce memory')
	cmd:option('-manualSeed', 0,        'Manually set RNG seed')
	cmd:option('-LR',         0.1,      'Initial learning rate')
	cmd:option('-momentum',   0.9,      'Momentum')
	cmd:option('-weightDecay',1e-4,     'Weight decay')
	-- Fine-tune
	cmd:option('-ft',         'false',  'Reinit the classifer for FT')
	cmd:option('-epochNumber',1,        'Manual epoch number (useful on restarts)')
	cmd:option('-retrain',    'none',   'Path to the model to retrain with')
	cmd:option('-nClasses',   0,        'Number of classes for FT datasets')
	cmd:option('-lockEpoch'  ,0,        'Number of Epoachs to lock Per-trained features')

	-- Test
	cmd:option('-tenCrop',    'false',   'Ten-crop testing')
	cmd:option('-testOnly',   'false',   'Only testing')

	cmd:text()

	local opt = cmd:parse(arg or {})

	opt.shareGradInput = opt.shareGradInput ~= 'false'
	opt.bottleneck = opt.bottleneck ~= 'false'
	opt.ft = opt.ft ~= 'false'
	opt.tenCrop = opt.tenCrop ~= 'false'
	opt.testOnly = opt.testOnly ~= 'false'
	opt.multisize = opt.multisize~= 'false'
	opt.threesize = opt.threesize~= 'false'

	if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
		cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
	end

	if opt.dataset == 'cifar10' then
		-- Default shortcutType=A and nEpochs=164
		opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
		opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
	elseif opt.dataset == 'stl10' then
		-- Default shortcutType=A and nEpochs=164
		opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
		opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
	elseif opt.dataset == 'joint' then
		-- Default shortcutType=A and nEpochs=164
		opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
		opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
	elseif opt.dataset == 'minc' then
		-- add the customize dataset here
		-- Handle the most common case of missing -data flag
		local trainDir = paths.concat(opt.data, 'images')
		if not paths.dirp(opt.data) then
			cmd:error('error: missing MINC data directory')
		elseif not paths.dirp(trainDir) then
			cmd:error('error: MINC missing `train` directory: ' .. trainDir)
		end
		-- Default shortcutType=B and nEpochs=90
		opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
		if opt.ft then
			opt.nEpochs = opt.nEpochs == 0 and 60 or opt.nEpochs
		else
			opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
		end
	
	else
		cmd:error('unknown dataset: ' .. opt.dataset)
	end

	return opt
end

return M
