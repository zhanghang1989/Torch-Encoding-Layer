--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Created by: Hang Zhang
-- ECE Department, Rutgers University
-- Email: zhang.hang@rutgers.edu
-- Copyright (c) 2017
--
-- Free to reuse and distribute this software for research or 
-- non-profit purpose, subject to the following conditions:
--  1. The code must retain the above copyright notice, this list of
--     conditions.
--  2. Original authors' names are not deleted.
--  3. The authors' names are not used to endorse or promote products
--      derived from this software 
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

local NetVLAD, parent = torch.class('nn.NetVLAD', 'nn.Module')

local function isint(x) 
	return type(x) == 'number' and x == math.floor(x) 
end

function NetVLAD:__init(K, D)
	parent.__init(self)
	-- assertions
	assert(self and K and D, 'should specify K and D')
	assert(isint(K) and isint(D), 'K and D should be integers')
	self.K = K
	self.D = D
	-- the dictionary, assigning matrix and residuals
	self.weight = torch.Tensor(K, D)
	self.A = torch.Tensor()
	self.R = torch.Tensor()
	-- the assigning drops the link with centers and
	-- is simplified as 1x1 conv with the input
	self.assigner = nn.Sequential()
	self.assigner:add(nn.Linear(D, K, false))
	self.assigner:add(nn.SoftMax())
	-- the gradient parameters
	self.gradInput = torch.Tensor()
	self.gradWeight = torch.Tensor(K, D)
	self.gradA = torch.Tensor()
	-- init the dictionary
	self:reset()
end

function NetVLAD:reset(stdv)
	if stdv then
		stdv = stdv * math.sqrt(3)
	else
		stdv = 1./math.sqrt(self.weight:size(2))
	end
	self.weight:uniform(-stdv,stdv)
	self.assigner:reset()
	return self
end

function NetVLAD:updateOutput(input)
	assert(self)
	assert(input:dim()==2 or input:dim()==3, 'only 2D or 3D input supported')
	-- calculate the A and R
	-- X \in R^{[Bx]NxD}
	local K = self.K
	local D = self.D

	if input:dim() == 2 then
		local N = input:size(1)
		-- assigning
		self.A = self.assigner:forward(input)
		-- calculate residuals
		self.R =  input:view(N,1,D):expand(N,K,D) 
							- self.weight:view(1,K,D):expand(N,K,D) 
	elseif input:dim() == 3 then
		B = input:size(1)
		local N = input:size(2)
		-- assigning
		self.A = self.assigner:forward(input:view(B*N, D)):view(B,
							N, K)
		-- calculate residuals
		self.R =  input:view(B,N,1,D):expand(B,N,K,D) 
							- self.weight:view(1,1,K,D):expand(B,N,K,D) 
	end
	
	if input:dim() == 2 then
		self.output:resize(K, D)
		HZENCODING.Aggregate.Forward(self.output:view(1,K,D), 
										self.A:view(1,N,K),
										self.R:view(1,N,K,D))
	elseif input:dim() == 3 then
		local B = self.A:size(1)
		self.output:resize(B, K, D)
		HZENCODING.Aggregate.Forward(self.output, self.A, self.R)
	end
	return self.output
end

function NetVLAD:updateGradInput(input, gradOutput)
	assert(self)
	assert(self.gradInput)	
	-- TODO assert the gtadOutput size
  -- N may vary during the training   
	self.gradA:resizeAs(self.A):fill(0)
	self.gradInput:resizeAs(input):fill(0)

	if self.A:dim() == 2 then
		-- d_l/d_A \in R^{NxK}
		local N = A:size(1)
		HZENCODING.Aggregate.BackwardA(self.gradA:view(1,N,K), 
												gradOutput:view(1,K,D), self.R:view(1,N,K,D))
		-- d_l/d_X = d_l/d_A * d_A/d_X + d_l/d_R * d_R/d_X
		self.gradInput = self.assigner:updateGradInput(input, self.gradA) 
											+ self.A * gradOutput		
	elseif self.A:dim() == 3 then
		local B = self.A:size(1)
		local N = input:size(2)
		-- d_l/d_A \in R^{NxK}
		HZENCODING.Aggregate.BackwardA(self.gradA, gradOutput, self.R)
		-- d_l/d_X = d_l/d_A * d_A/d_X + d_l/d_R * d_R/d_X
		self.gradInput= self.assigner:updateGradInput(input:view(B*N,self.D), 
										self.gradA:view(B*N,self.K))
										:view(B,N,self.D) 
										+ torch.bmm(self.A, gradOutput)
	else
		error('input must be 2D or 3D')
	end

	return self.gradInput
end

function NetVLAD:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	-- update the assigner
	self.assigner:accUpdateGradParameters(input, self.gradA, scale)
	-- update the dictionary
	if self.A:dim() == 2 then
		for k = 1,self.K do
			-- d_l/d_c 
			self.gradWeight[k] = -scale*self.A:select(2,k):sum() * gradOutput[k] 
		end
	elseif self.A:dim() == 3 then
		local B = self.A:size(1)
		for b = 1,B do
			for k = 1,self.K do
				-- d_l/d_c 
				self.gradWeight[k] = self.gradWeight[k] 
										-scale * self.A[b]:select(2,k):sum() * gradOutput[b][k]
			end
		end
	end
end

function NetVLAD:__tostring__()
	return torch.type(self) ..
		string.format(
			'(Nx%d -> %dx%d)',
			self.D, self.K, self.D
		)
end

function NetVLAD:cuda()
	self.assigner:cuda()
	return self.cuda()
end

function NetVLAD:training()
	self.assigner:training()
	return self
end

function NetVLAD:evaluation()
	self.assigner:evaluation()
	return self
end
