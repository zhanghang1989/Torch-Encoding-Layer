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

local Encoding, parent = torch.class('nn.Encoding', 'nn.Module')

local function isint(x) 
	return type(x) == 'number' and x == math.floor(x) 
end

function Encoding:__init(K, D)
	parent.__init(self)
	-- assertions
	assert(self and K and D, 'should specify K and D')
	assert(isint(K) and isint(D), 'K and D should be integers')
	self.K = K
	self.D = D
	-- the dictionary
	self.weight = torch.Tensor(K, D)
	-- the assigning factors (smoothing)
	self.bias = torch.Tensor(K)
	-- the assigning matrix and residuals
	self.A = torch.Tensor()
	self.R = torch.Tensor()
	-- the soft assigning operation
	self.soft = nn.SoftMax()
	self.batchMul = nn.MM()
	-- the gradient parameters
	self.gradInput = torch.Tensor()
	self.gradWeight = torch.Tensor(K, D)
	self.gradBias = torch.Tensor(K):abs()
	self.gradA = torch.Tensor()
	-- init the dictionary
	self:reset()
end

function Encoding:reset(stdv)
	local stdv1, stdv2
	if stdv then
		stdv1 = stdv * math.sqrt(3)
		stdv2 = stdv * math.sqrt(3)
	else
		stdv1 = 1./math.sqrt(self.weight:size(1))
		stdv2 = 1./math.sqrt(self.bias:size(1))
	end
	self.weight:uniform(-stdv1,stdv1)
	self.bias:uniform(-stdv2,stdv2)
	return self
end

function Encoding:updateOutput(input)
	assert(self)
	assert(input:dim()==2 or input:dim()==3, 'only 2D or 3D input supported')
	-- lazy init for weighted L2
	self.L2 = self.L2 or self.A.new()
	self.SL2 = self.SL2 or self.A.new()
	-- calculate the A and R
	-- X \in R^{[Bx]NxD}
	local K = self.K
	local D = self.D
	local N
	if input:dim() == 2 then
		N = input:size(1)
		-- calculate residuals
		self.R =  input:view(N,1,D):expand(N,K,D) 
							- self.weight:view(1,K,D):expand(N,K,D) 
		-- L2 norm of r_ik (assuming the N and K > 1)
		self.L2 = self.R:clone()
		self.L2 = self.L2:pow(2):sum(3):squeeze()
		-- weighted 
		self.SL2 = - self.L2 * self.bias:diag()
		self.A = self.soft:forward(self.SL2)
	elseif input:dim() == 3 then
		local B = input:size(1)
		N = input:size(2)
		-- calculate residuals
		self.R:resize(B,N,K,D)
		self.A:resize(B,N,K)
		self.SL2:resize(B,N,K)

		self.R:copy( input:view(B,N,1,D):expand(B,N,K,D) 
							- self.weight:view(1,1,K,D):expand(B,N,K,D) )
		-- L2 norm of r_ik
		self.L2 = self.R:clone()
		self.L2 = self.L2:pow(2):sum(4):view(B,N,K)
		-- weighted 
		self.SL2:copy( -torch.bmm(
			self.L2,	self.bias:diag():view(1,K,K):expand(B,K,K)) )
		self.A:copy( self.soft:forward(
			self.SL2:view(B*N,K)
			):view(B,N,K) )
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

function Encoding:updateGradInput(input, gradOutput)
	assert(self)
	assert(self.gradInput)	
	assert(input:dim()==2 or input:dim()==3, 'only 2D or 3D input supported')
  -- N may vary during the training   
	local K = self.K
	local D = self.D
	self.gradA:resizeAs(self.A):fill(0)
	self.gradInput:resizeAs(input):fill(0)
	self.gradSL2 = self.gradSL2 or self.gradA.new()

	if input:dim() == 2 then
		-- d_l/d_A \in R^{NxK}
		local N = A:size(1)
		HZENCODING.Aggregate.BackwardA(self.gradA:view(1,N,K), 
												gradOutput:view(1,K,D), self.R:view(1,N,K,D))
		-- d_l/d_X = d_l/d_A * d_A/d_X + d_l/d_R * d_R/d_X
		self.gradSL2 = self.soft:updateGradInput(self.SL2, self.gradA)
		self.gradInput:copy( 2*torch.bmm(
			(self.gradSL2 * self.bias:diag()):view(N,1,K),
			self.R):squeeze()
			+ self.A * gradOutput )
	elseif input:dim() == 3 then
		local B = self.A:size(1)
		local N = input:size(2)
		-- d_l/d_A \in R^{NxK}
		HZENCODING.Aggregate.BackwardA(self.gradA, gradOutput, self.R)
		-- d_l/d_X = d_l/d_A * d_A/d_X + d_l/d_R * d_R/d_X
		self.gradSL2 = self.soft:updateGradInput(self.SL2:view(B*N,K), 
														self.gradA:view(B*N,K)):view(B,N,K)
		self.gradInput:copy( 2*torch.bmm(
				(self.gradSL2:view(B*N,K)*self.bias:diag()):view(B*N,1,K), 
				self.R:view(B*N,K,D)):view(B,N,D)
				+ torch.bmm(self.A, gradOutput) )
	end
	return self.gradInput
end

function Encoding:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	local K = self.K
	local D = self.D
	self.bufBias = self.bufBias or self.bias.new()
	self.bufWeight = self.bufWeight or self.weight.new()
	
	if input:dim() == 2 then
		local N = input:size(1)
		-- d_loss/d_C = d_loss/d_R * d_R/d_C + d_loss/d_A * d_A/d_C
		HZENCODING.Weighting.UpdateParams(self.gradBias:view(1,K), 
						self.gradSL2:view(1,N,K), self.L2:view(1,N,K))
		for k = 1,self.K do
				-- d_l/d_c 
				self.gradWeight[k] = -scale*self.A:select(2,k):sum()*gradOutput[k]
										-2*scale*self.gradSL2:select(2,k):reshape(1,N) 
											* self.bias[k] * self.R:select(2,k)
		end
	elseif input:dim() == 3 then
		local B = self.A:size(1)
		local N = input:size(2)
		-- batch gradient of s_k
		self.bufBias:resize(B, K)
		HZENCODING.Weighting.UpdateParams(self.bufBias, self.gradSL2, self.L2)
		-- average the gradient for s in the batch instead of sum, avoid overflow
		self.gradBias:copy(scale * self.bufBias:sum(1):squeeze())

		self.bufWeight:resize(B,K,D)
		HZENCODING.Weighting.BatchRowScale(self.bufWeight, self.A:sum(2):squeeze(), gradOutput)
		self.gradWeight:copy( -2*scale*	torch.bmm(
				(self.gradSL2:view(B*N,K)*self.bias:diag()):view(B,N,K):transpose(2,3):reshape(B*K,1,N),
				self.R:transpose(2,3):reshape(B*K,N,D)):view(B,K,D):sum(1):squeeze() 
			-scale * self.bufWeight:sum(1):squeeze())
	end
end

function Encoding:__tostring__()
	return torch.type(self) ..
		string.format(
			'(Nx%d -> %dx%d)',
			self.D, self.K, self.D
		)
end

