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

local Aggregate, parent = torch.class('nn.Aggregate', 'nn.Module')

local function isint(x) 
	return type(x) == 'number' and x == math.floor(x) 
end

function Aggregate:__init(K, D)
	parent.__init(self)
	-- assertions
	assert(self and K and D, 'should specify K and D')
	assert(isint(K) and isint(D), 'K and D should be integers')
	self.K = K
	self.D = D
	self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function Aggregate:updateOutput(input)
	assert(self)
	local K = self.K
	local D = self.D
	A = input[1]
	R = input[2]
	-- TODO assert inputs (A \in R^{NxK} R \in R^{NxKxD})

	if A:dim() == 2 then
		self.output:resize(K, D)
		-- aggregation operation (in Matrix From)
		-- e_k = a_k^T * R_k, where a_k and R_k are expanded at 2nd dim
		for k=1,self.K do
			self.output:select(1, k):copy(torch.mv(R:select(2, k):t(), 
																	A:select(2, k)))
		end
	elseif A:dim() == 3 then
		local B = A:size(1)
		self.output:resize(B, K, D)
		for b=1, B do 
			for k=1,self.K do
				self.output[b]:select(1,k):copy(torch.mv(R[b]:select(2, k):t(), 
																	A[b]:select(2, k)))
			end
		end
	else
		error('input must be 2D or 3D')
	end
	return self.output
end

function Aggregate:updateGradInput(input, gradOutput)
	assert(self)
	assert(self.gradInput)	
	A = input[1]
	R = input[2]

	-- TODO assert the gtadOutput size
	if #self.gradInput == 0 then
  	for i = 1, 2 do self.gradInput[i] = input[i].new() end
  end

  -- N may vary during the training   
	self.gradInput[1]:resizeAs(input[1]):fill(0)
  self.gradInput[2]:resizeAs(input[2]):fill(0)
	

	if A:dim() == 2 then
		-- d_l/d_A \in R^{NxK}
		for k = 1,self.K do
			-- d_l/d_a_k = R_k * d_l/d_e_k
			self.gradInput[1]:select(2,k):copy(
				torch.mv(R:select(2, k), gradOutput[k])
			)
		end
		-- d_l/d_R \in R^{NxKxD}
		for k = 1,self.K do
			-- d_l/d_R_k = a_k * {d_l/d_e_k}^T
			self.gradInput[2]:select(2,k):addr(
				A:select(2,k),gradOutput[k]
			)
		end
	elseif A:dim() == 3 then
		local B = A:size(1)
		-- d_l/d_A \in R^{NxK}
		for b=1, B do 
			for k = 1,self.K do
				-- d_l/d_a_k = R_k * d_l/d_e_k
				self.gradInput[1][b]:select(2,k):copy(
					torch.mv(R[b]:select(2, k), gradOutput[b][k])
				)
			end
			-- d_l/d_R \in R^{NxKxD}
			for k = 1,self.K do
				-- d_l/d_R_k = a_k * {d_l/d_e_k}^T
				self.gradInput[2][b]:select(2,k):addr(
					A[b]:select(2,k),gradOutput[b][k]
				)
			end
		end
	else
		error('input must be 2D or 3D')
	end

	return self.gradInput
end

function Aggregate:__tostring__()
	return torch.type(self) ..
		string.format(
			'(Nx%d, Nx%dx%d -> %dx%d)',
			self.K, self.K, self.D, self.K, self.D
		)
end

