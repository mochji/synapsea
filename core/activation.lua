--[[
	https://github.com/x-xxoa/synapsea
	core/activation.lua

	MIT License
]]--

local activation = {
	sigmoid,
	tanh,
	relu,
	leakyrelu,
	elu,
	exponential,
	swish,
	binarystep,
	softmax,
	softplus,
	softsign,
	linear,
	variablelinear,
	hardsigmoid,
	hardtanh
}

function activation.sigmoid(x, derivative)
	if derivative then
		return (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x))))
	end

	return 1 / (1 + math.exp(-x))
end

function activation.tanh(x, derivative)
	if derivative then
		return 1 - (((math.exp(2 * x) - 1) / (math.exp(2 * x) + 1))^2)
	end

	return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
end

function activation.relu(x, derivative)
	if derivative then
		if x > 0 then
			return 1
		end

		return 0
	end

	return math.max(0, x)
end

function activation.leakyrelu(x, derivative, alpha)
	if derivative then
		if x > 0 then
			return 1
		end

		return alpha
	end

	if x > 0 then
		return x
	end

	return x * alpha
end

function activation.elu(x, derivative, alpha)
	if derivative then
		if x < 0 then
			return alpha * math.exp(x)
		end

		return 1
	end

	if x < 0 then
		return alpha * (math.exp(x) - 1)
	end

	return x
end

function activation.exponential(x)
	return math.exp(x)
end

function activation.swish(x, derivative, alpha)
	if derivative then
		return (math.exp(-alpha * x) * x + math.exp(-alpha * x) + 1) / ((math.exp(-alpha * x) + 1)^2)
	end

	return x / (1 + math.exp(-alpha * x))
end

function activation.binarystep(x, derivative)
	if derivative then
		return 0
	end

	if x > 0 then
		return 1
	end

	return 0
end

function activation.softmax(x, derivative)
	local expSum, output = 0, {}

	for a = 1, #x do
		expSum = expSum + math.exp(x[a])
	end

	if derivative then
		for a = 1, #x do
			output[a] = (math.exp(x[a]) / expSum) * (1 - (math.exp(x[a]) / expSum))
		end

		return output
	end

	for a = 1, #x do
		output[a] = math.exp(x[a]) / expSum
	end

	return output
end

function activation.softplus(x, derivative)
	if derivative then
		return 1 / (1 + math.exp(-x))
	end

	return math.log(1 + math.exp(x))
end

function activation.softsign(x, derivative)
	if derivative then
		if x == 0 then
			return 1 -- at x = 0 this is undefined so this is a fix
		end

		return x / (x * (1 + math.abs(x)^2))
	end

	return x / (1 + math.abs(x))
end

function activation.linear(x, derivative)
	if derivative then
		return 1
	end

	return x
end

function activation.variablelinear(x, derivative, alpha)
	if derivative then
		return alpha
	end

	return x * alpha
end

function activation.hardsigmoid(x, derivative)
	if derivative then
		if x < -2.5 or x > 2.5 then
			return 0
		end

		return 0.2
	end

	return math.max(0, math.min(1, x * 0.2 + 0.5))
end

function activation.hardtanh(x, derivative)
	if derivative then
		if x < -1 or x > 1 then
			return 0
		end

		return 1
	end

	return math.max(0, math.min(1, x * 2))
end

return activation
