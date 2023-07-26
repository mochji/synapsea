--[[
	https://github.com/x-xxoa/synapsea
	core/backprop.lua

	MIT License
]]--

-- error = (weight_k * error_j) * act'(output)

local activation = require("core.activation")
local loss = require("core.loss")
local syntable = require("core.syntable")
local backprop = {}

function backprop.outputError(args)
	local err = {}

	local activation, loss = activation[args.activation], loss[args.loss]

	for a = 1, #args.output do
		if type(args.output[a]) == "table" then
			err[a] = backprop.outputError{
				output = args.output[a],
				expectedOutput = args.expectedOutput[a],
				activation = args.activation,
				alpha = args.alpha,
				loss = args.loss,
				activationArgs = args.activationArgs,
				lossArgs = args.lossArgs
			}
		else
			err[a] = loss(args.output[a], args.expectedOutput[a], args.lossArgs) * activation(args.output[a], true, args.alpha, args.activationArgs)
		end
	end

	return err
end

function backprop.prelu(args)
	local sum = 0

	local activation = activation[args.activation]
	args.nextError = syntable.flatten(args.nextError)


	for a = 1, #args.nextError do
		sum = sum + args.nextError[a] * (activation(args.nextError[a], false, args.alpha, args.activationArgs) / args.alpha)
	end

	return sum
end

function backprop.dense(args)
	local err = {}

	local activation = activation[args.activation]
	args.bias = args.bias or 0

	for a = 1, #args.output do
		local sum = 0
	
		for b = 1, #args.nextError do
			sum = sum + args.weights[a][b] * activation(args.output[a], true, args.alpha, args.activationArgs)
		end

		err[a] = sum
	end

	return err
end

function backprop.convolutional1D(args)
	local err = {}

	local activation = activation[args.activation]

	for a = 1, #args.output do
		local sum = 0

		for b = 1, #args.nextError do
			for c = 1, #args.filter do
				if c % args.dilation[1] == 0 then
					sum = sum + (args.filter[c] * args.nextError[b]) * activation(args.output[a], true, args.alpha, args.activationArgs)
				end
			end
		end

		err[a] = sum
	end

	return err
end

function backprop.convolutional2D(args)
	local err = {}

	local activation = activation[args.activation]

	for a = 1, #args.output do
		for b = 1, #args.output[a] do
			local sum = 0

			for c = 1, #args.nextError do
				for d = 1, #args.nextError[c] do
					for e = 1, #args.filter do
						for f = 1, #args.filter[e] do
							if e % args.dilation[1] == 0 and f % args.dilation[2] == 0 then
								sum = sum + (args.filter[e][f] * args.nextError[c][d]) * activation(args.output[a][b], true, args.alpha, args.activationArgs)
							end
						end
					end
				end
			end

			err[a][b] = sum
		end
	end

	return err
end

function backprop.convolutional3D(args)
	local err = {}

	local activation = activation[args.activation]

	for a = 1, #args.output do
		for b = 1, #args.output[a] do
			for c = 1, #args.output[a][b] do
				local sum = 0

				for d = 1, #args.nextError do
					for e = 1, #args.nextError[d] do
						for f = 1, #args.nextError[d][e] do
							for g = 1, #args.filter do
								for h = 1, #args.filter[f] do
									for i = 1, #args.filter[f][h] do
										if g % args.dilation[1] == 0 and h % args.dilation[2] == 0 and i % args.dilation[3] == 0 then
											sum = sum + (args.filter[g][h][i] * args.nextError[d][e][f]) * activation(args.output[a][b][c], true, args.alpha, args.activationArgs)
										end
									end
								end
							end
						end
					end
				end

				err[a][b][c] = sum
			end
		end
	end

	return err
end

return backprop
