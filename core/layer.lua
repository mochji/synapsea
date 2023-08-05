--[[
	https://github.com/x-xxoa/synapsea
	core/layer.lua

	MIT License
]]--

local activation = require("core.activation")
local syntable = require("core.syntable")
local layer = {
	dense,
	averagePooling1D,
	averagePooling2D,
	averagePooling3D,
	maxPooling1D,
	maxPooling2D,
	maxPooling3D,
	sumPooling1D,
	sumPooling2D,
	sumPooling3D,
	averageGlobalPooling1D,
	averageGlobalPooling2D,
	averageGlobalPooling3D,
	maxGlobalPooling1D,
	maxGlobalPooling2D,
	maxGlobalPooling3D,
	sumGlobalPooling1D,
	sumGlobalPooling2D,
	sumGlobalPooling3D,
	upSample1D,
	upSample2D,
	upSample3D,
	zeroPad1D,
	zeroPad2D,
	zeroPad3D,
	crop1D,
	crop2D,
	crop3D,
	randomCrop1D,
	randomCrop2D,
	randomCrop3D,
	convolutional1D,
	convoltuional2D,
	convolutional3D,
	convolutionalTranspose1D,
	convolutionalTranspose2D,
	convolutionalTranspose3D,
	convolutionalDepthwise1D,
	convolutionalDepthwise2D,
	convolutionalSeparable1D,
	convolutionalSeparable2D,
	convolutionalSeparable3D,
	convolutionalDepthwiseSeparable1D,
	convolutionalDepthwiseSeparable2D,
	locallyConnected1D,
	locallyConnected2D,
	locallyConnected3D,
	flatten,
	normalize1D,
	normalize2D,
	normalize3D,
	vectorAdd1D,
	vectorAdd2D,
	vectorAdd3D,
	vectorSubtract1D,
	vectorSubtract2D,
	vectorSubtract3D,
	dot1D,
	dot2D,
	dot3D,
	vectorDivide1D,
	vectorDivide2D,
	vectorDivide3D,
	dropOut,
	uniformNoise,
	normalNoise,
	softmax,
	activate
}

function layer.dense(args)
	local output = {}

	local activation = activation[args.activation]
	args.bias = args.bias or 0

	for a = 1, args.outputSize do
		local sum = 0

		for b = 1, #args.input do
			sum = sum + args.input[b]*args.weights[b][a]
		end

		output[a] = activation(sum + args.bias, false, args.alpha)
	end

	return output
end

function layer.averagePooling1D(args)
	local output, startIndex = {}, 0 -- even though its named startIndex its actually startIndex - 1 but startIndexMinusOne isnt a good variable name and idrk what else to call it :/
	local outputSize = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1

	for a = 1, outputSize do
		local sum = 0

		for b = 1, #args.kernel[1] do
			if b % args.dilation[1] == 0 then
				sum = sum + args.input[startIndex + b]
			end
		end

		start = start + args.stride[1]

		output[a] = sum / args.kernel[1]
	end

	return output
end

function layer.averagePooling2D(args)
	local output, startIndexA, kernelProduct = {}, 0, syntable.product(args.kernel) -- were getting the product of the kernel so we dont perform the same calculation every output node
	local outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputWidth do
			local sum = 0

			for c = 1, args.kernel[1] do
				for d = 1, args.kernel[2] do
					if c % args.dilation[1] == 0 and d & args.dilation[2] == 1 then
						sum = sum + args.input[startIndexA + c][startIndexB + d]
					end
				end
			end

			startIndexB = startIndexB + args.stride[2]

			output[a][b] = sum / kernelProduct
		end

		startIndexA = startIndexA + args.stride[1]
	end

	return output
end

function layer.averagePooling3D(args)
	local output, startIndexA, kernelProduct = {}, 0, syntable.product(args.kernel)
	local outputDepth, outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1, math.floor((#args.input[1][1] - args.kernel[3]) / args.stride[3]) + 1

	for a = 1, outputDepth do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputHeight do
			output[a][b] = {}
			local startIndexC = 0

			for c = 1, outputWidth do
				local sum = 0

				for d = 1, args.kernel[1] do
					for e = 1, args.kernel[2] do
						for f = 1, args.kernel[3] do
							if d % args.dilation[1] == 0 and e % args.dilation[2] == 0 and f % args.dilation[3] == 0 then
								sum = sum + args.input[startIndexA + d][startIndexB + e][startIndexC + f]
							end
						end
					end
				end

				startIndexC = startIndexC + args.stride[3]

				output[a][b][c] = sum / kernelProduct
			end

			startIndexB = startIndexB + args.stride[2]
		end

		startIndexA = startIndexA + args.stride[1]
	end

	return output
end

function layer.maxPooling1D(args)
	local output, startIndex = {}, 0
	local outputSize = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1

	for a = 1, outputSize do
		local max = args.input[startIndex + 1]

		for b = 1, #args.kernel[1] do -- yes i know we're comparing the first one against itself in the first iteration of this loop but its not really that much and 1 per output node is probably fine
			if b % args.dilation[1] == 0 then
				max = math.max(max, args.input[startIndex + b])
			end
		end

		start = start + args.stride[1]

		output[a] = max
	end

	return output
end

function layer.maxPooling2D(args)
	local output, startIndexA = {}, 0
	local outputHeight, outputWidth = math.floor((#args.filter - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.filter[1] - args.kernel[2]) / args.stride[2]) + 1

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputWidth do
			local max = args.input[startIndexA + 1][startIndexB + 1]

			for c = 1, args.kernel[1] do
				for d = 1, args.kernel[2] do
					if c % args.dilation[1] == 0 and d % args.dilation[2] == 0 then
						max = math.max(max, args.input[startIndexA + c][startIndexB + d])
					end
				end
			end

			startIndexB = startIndexB + args.stride[2]

			output[a][b] = max
		end

		startIndexA = startIndexA + args.stride[1]
	end

	return output
end

function layer.maxPooling3D(args)
	local output, startIndexA = {}, 0
	local outputDepth, outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1, math.floor((#args.input[1][1] - args.kernel[3]) / args.stride[3]) + 1

	for a = 1, outputDepth do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputHeight do
			output[a][b] = {}
			local startIndexC = 0

			for c = 1, outputWidth do
				local max = args.input[startIndexA + 1][startIndexB + 1][startIndexC + 1]

				for d = 1, args.kernel[1] do
					for e = 1, args.kernel[2] do
						for f = 1, args.kernel[3] do
							if d % args.dilation[1] == 0 and e % args.dilation[2] == 0 and f % args.dilation[3] == 0 then
								max = math.max(max, args.input[startIndexA + d][startIndexB + e][startIndexC + f])
							end
						end
					end
				end

				startIndexC = startIndexC + args.stride[3]

				output[a][b][c] = max
			end

			startIndexB = startIndexB + args.stride[2]
		end

		startIndexA = startIndexA + args.stride[1]
	end

	return output
end

function layer.sumPooling1D(args)
	local output, startIndex = {}, 0
	local outputSize = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1

	for a = 1, outputSize do
		local sum = 0

		for b = 1, #args.kernel[1] do
			if b % args.dilation[1] == 0 then
				sum = sum + args.input[startIndex + b]
			end
		end

		start = start + args.stride[1]

		output[a] = sum
	end

	return output
end

function layer.sumPooling2D(args)
	local output, startIndexA = {}, 0
	local outputHeight, outputWidth = math.floor((#args.filter - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.filter[1] - args.kernel[2]) / args.stride[2]) + 1

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputWidth do
			local sum = 0

			for c = 1, args.kernel[1] do
				for d = 1, args.kernel[2] do
					if c % args.dilation[1] == 0 and d & args.dilation[2] == 1 then
						sum = sum + args.input[startIndexA + c][startIndexB + d]
					end
				end
			end

			startIndexB = startIndexB + args.stride[2]

			output[a][b] = sum
		end

		startIndexA = startIndexA + args.stride[1]
	end

	return output
end

function layer.sumPooling3D(args)
	local output, startIndexA = {}, 0
	local outputDepth, outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1, math.floor((#args.input[1][1] - args.kernel[3]) / args.stride[3]) + 1

	for a = 1, outputDepth do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputHeight do
			output[a][b] = {}
			local startIndexC = 0

			for c = 1, outputWidth do
				local sum = 0

				for d = 1, args.kernel[1] do
					for e = 1, args.kernel[2] do
						for f = 1, args.kernel[3] do
							if d % args.dilation[1] == 0 and e % args.dilation[2] == 0 and f % args.dilation[3] == 0 then
								sum = sum + args.input[startIndexA + d][startIndexB + e][startIndexC + f]
							end
						end
					end
				end

				startIndexC = startIndexC + args.stride[3]

				output[a][b][c] = sum
			end

			startIndexB = startIndexB + args.stride[2]
		end

		startIndexA = startIndexA + args.stride[1]
	end

	return output
end

function layer.averageGlobalPooling1D(args)
	return {syntable.sum(args.input) / #args.input}
end

function layer.averageGlobalPooling2D(args)
	local output = {}

	for i = 1, #args.input do
		output[i] = syntable.sum(args.input[i]) / #args.input
	end

	return output
end

layer.averageGlobalPooling3D = layer.averageGlobalPooling2D -- syntable.sum() is recursive so doing this works

function layer.maxGlobalPooling1D(args)
	return {syntable.max(args.input)}
end

function layer.maxGlobalPooling2D(args)
	local output = {}

	for i = 1, #args.input do
		output[i] = syntable.max(args.input[i])
	end

	return output
end

layer.maxGlobalPooling3D = layer.maxGlobalPooling2D

function layer.sumGlobalPooling1D(args)
	return {syntable.sum(args.input)}
end

function layer.sumGlobalPooling2D(args)
	local output = {}

	for i = 1, #args.input do
		output[i] = syntable.sum(args.input[i])
	end

	return output
end

layer.sumGlobalPooling3D = layer.sumGlobalPooling2D

function layer.upSample1D(args)
	local output = {}

	for a = 1, #args.input * args.kernel[1] do
		output[a] = args.input[math.ceil(i / args.kernel[1])]
	end

	return output
end

function layer.upSample2D(args)
	local output = {}

	for a = 1, #args.input * args.kernel[1] do
		output[a] = {}

		for b = 1, #args.input[1] * args.kernel[2] do
			output[a][b] = args.input[math.ceil(a / args.kernel[1])][math.ceil(b / args.kernel[2])]
		end
	end

	return output
end

function layer.upSample3D(args)
	local output = {}

	for a = 1, #args.input * args.kernel[1] do
		output[a] = {}

		for b = 1, #args.input * args.kernel[2] do
			output[a][b] = {}

			for c = 1, #args.input*args.kernel[3] do
				output[a][b][c] = args.input[math.ceil(a / args.kernel[1])][math.ceil(b / args.kernel[2])][math.ceil(c / args.kernel[3])]
			end
		end
	end

	return output
end

function layer.zeroPad1D(args)
	for a = 1, args.paddingAmount[1] do
		table.insert(args.input, 1, 0)

		args.input[#args.input + 1] = 0
	end

	return args.input
end

function layer.zeroPad2D(args)
	-- pad the tops and bottoms

	for a = 1, args.paddingAmount[1] do
		table.insert(args.input, 1, {})

		for b = 1, #args.input[a + 1] do
			args.input[1][b] = 0
		end

		args.input[#args.input + 1] = args.input[1]
	end

	-- pad the sides

	for a = 1, #args.input do
		for b = 1, args.paddingAmount[2] do
			table.insert(args.input[a], 1, 0)

			args.input[a][#args.input[a] + 1] = 0
		end
	end

	return args.input
end

function layer.zeroPad3D(args)
	for a = 1, #args.input do
		args.input[a] = layer.zeroPad2D{
			input = args.input[a],
			paddingAmount = {args.paddingAmount[2], args.paddingAmount[3]}
		}
	end

	for a = 1, args.paddingAmount[1] do
		table.insert(args.input, 1, {})
		for b = 1, #args.input[a + 1] do
			args.input[1][b] = {}
			for c = 1, #args.input[a + 1][b] do
				args.input[1][b][c] = 0
			end
		end

		args.input[#args.input + 1] = args.input[1]
	end

	return args.input
end

function layer.crop1D(args)
	local output = {}

	for a = 1, args.start[1], args.outputShape[1] + args.start[1] do
		output[#output + 1] = args.input[a]
	end

	return output
end

function layer.crop2D(args)
	local output = {}

	for a = 1, args.start[1], args.outputShape[1] + args.start[1] do
		local row = {}

		for b = 1, args.start[2], args.outputShape[2] + args.start[2] do
			row[#row + 1] = args.input[a][b]
		end

		output[#output + 1] = row
	end

	return output
end

function layer.crop3D(args)
	local output = {}

	for a = 1, args.start[1], args.outputShape[1] + args.start[1] do
		local crop = {}

		for b = 1, args.start[2], args.outputShape[2] + args.start[2] do
			local row = {}

			for c = 1, args.start[3], args.outputShape[3] + args.start[3] do
				row[#row + 1] = args.input[a][b][c]
			end

			crop[#crop + 1] = row
		end

		output[#output + 1] = crop
	end

	return output
end

function layer.randomCrop1D(args)
	return layer.crop1D{
		input = args.input,
		outputShape = args.outputShape,
		start = {math.random(1, args.outputShape[1])}
	}
end

function layer.randomCrop2D(args)
	return layer.crop2D{
		input = args.input,
		outputShape = args.outputShape,
		start = {
			math.random(1, args.outputShape[1]),
			math.random(1, args.outputShape[2])
		}
	}
end

function layer.randomCrop3D(args)
	return layer.crop3D{
		input = args.input,
		outputShape = args.outputShape,
		start = {
			math.random(1, args.outputShape[1]),
			math.random(1, args.outputShape[2]),
			math.random(1, args.outputShape[2])
		}
	}
end

function layer.convolutional1D(args)
	local output, startIndex = {}, 0
	local outputSize = math.floor((#args.input - #args.filter) / args.stride[1]) + 1

	local activation = activation[args.activation]

	for a = 1, outputSize do
		local sum = 0

		for b = 1, #args.filter do
			if b % args.dilation[1] == 0 then
				sum = sum + args.input[startIndex + b] * args.filter[b]
			end
		end

		startIndex = startIndex + args.stride[1]

		if args.biases then
			output[a] = activation(sum + args.biases[a], false, args.alpha)
		else
			output[a] = activation(sum, false, args.alpha)
		end
	end

	return output
end

function layer.convolutional2D(args)
	local output, startIndexA = {}, 0
	local outputHeight, outputWidth = math.floor((#args.input - #args.filter) / args.stride[1]) + 1, math.floor((#args.input[1] - #args.filter[1]) / args.stride[2]) + 1

	local activation = activation[args.activation]

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1,outputWidth do
			local sum = 0

			for c = 1, #args.filter do
				for d = 1, #args.filter[c] do
					if c % args.dilation[1] == 0 and d % args.dilation[2] == 0 then
						sum = sum + args.input[startIndexA + c][startIndexB + d] * args.filter[c][d]
					end
				end
			end

			startIndexB = startIndexB + args.stride[2]

			if args.biases then
				output[a][b] = activation(sum + args.biases[a][b], false, args.alpha)
			else
				output[a][b] = activation(sum, false, args.alpha)
			end
		end

		startIndexA = startIndexA + args.stride[1]
	end

	return output
end

function layer.convolutional3D(args)
	local output, startIndexA = {}, 0
	local outputDepth, outputHeight, outputWidth = math.floor((#args.input - #args.filter) / args.stride[1]) + 1, math.floor((#args.input[1] - #args.filter[1]) / args.stride[2]) + 1, math.floor((#args.input[1][1] - #args.filter[1][1]) / args.stride[3]) + 1

	local activation = activation[args.activation]

	for a = 1, outputDepth do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputHeight do
			output[a][b] = {}
			local startIndexC = 0

			for c = 1, outputWidth do
				local sum = 0

				for d = 1, #args.filter do
					for e = 1, #args.filter[d] do
						for f = 1, #args.filter[d][e] do
							if d % args.dilation[1] == 0 and e % args.dilation[2] == 0 and f % args.dilation[3] == 0 then
								sum = sum + args.input[startIndexA + d][startIndexB + e][startIndex + f] * args.filter[d][e][f]
							end
						end
					end
				end

				startIndexC = startIndexC + args.stride[3]

				if args.biases then
					output[a][b][c] = activation(sum + args.biases[a][b][c], false, args.alpha)
				else
					output[a][b][c] = activation(sum, false, args.alpha)
				end
			end

			startIndexB = startIndexB + args.stride[2]
		end

		startIndexA = startIndexA + args.stride[1]
	end

	return output
end

function layer.convolutionalTranspose1D(args)
	return layer.convolutional1D{
		input = layer.zeroPad1D(args.input, args.paddingAmount),
		filter = args.filter,
		biases = args.biases,
		dilation = args.dilation,
		activation = args.activation,
		alpha = args.alpha
	}
end

function layer.convolutionalTranspose2D(args)
	return layer.convolutional2D{
		input = layer.zeroPad2D(args.input, args.paddingAmount),
		filter = args.filter,
		biases = args.biases,
		dilation = args.dilation,
		activation = args.activation,
		alpha = args.alpha
	}
end

function layer.convolutionalTranspose3D(args)
	return layer.convolutional3D{
		input = layer.zeroPad3D(args.input, args.paddingAmount),
		filter = args.filter,
		biases = args.biases,
		dilation = args.dilation,
		activation = args.activation,
		alpha = args.alpha
	}
end

function layer.convolutionalDepthwise1D(args)
	local output = {}

	for a = 1, #args.input do
		local tempArgs = {
			input = args.input[a],
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutional1D(tempArgs)
	end

	return output
end

function layer.convolutionalDepthwise2D(args)
	local output = {}

	for a = 1, #args.input do
		local tempArgs = {
			input = args.input[a],
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutional2D(tempArgs)
	end

	return output
end

function layer.convolutionalSeparable1D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutional1D(tempArgs)
	end

	return output
end

function layer.convolutionalSeparable2D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutional2D(tempArgs)
	end

	return output
end

function layer.convolutionalSeparable3D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutional3D(tempArgs)
	end

	return output
end

function layer.convolutionalDepthwiseSeparable1D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutionalDepthwise1D(tempArgs)
	end

	return output
end

function layer.convolutionalDepthwiseSeparable2D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutionalDepthwise2D(tempArgs)
	end

	return output
end

function layer.flatten(args)
	return syntable.flatten(args.input)
end

function layer.normalize1D(args)
	local max, min = syntable.max(args.input), syntable.min(args.input)
	local maxMinusMin = max - min

	for a = 1, #args.input do
		args.input[a] = args.input[a] - min / maxMinusMin
	end

	return args.input
end

function layer.normalize2D(args)
	local max, min = syntable.max(args.input), syntable.min(args.input)
	local maxMinusMin = max - min

	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			args.input[a][b] = args.input[a][b] - min / maxMinusMin
		end
	end

	return args.input
end

function layer.normalize3D(args)
	local max, min = syntable.max(args.input), syntable.min(args.input)
	local maxMinusMin = max - min

	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			for c = 1, #args.input[a][b] do
				args.input[a][b][c] = args.input[a][b][c] - min / maxMinusMin
			end
		end
	end

	return args.input
end

function layer.vectorAdd1D(args)
	for a = 1, #args.input do
		args.input[a] = args.input[a] + args.biases[a]
	end

	return args.input
end

function layer.vectorAdd2D(args)
	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			args.input[a][b] = args.input[a][b] + args.biases[a][b]
		end
	end

	return args.input
end

function layer.vectorAdd3D(args)
	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			for c = 1, #args.input[a][b] do
				args.input[a][b][c] = args.input[a][b][c] + args.biases[a][b][c]
			end
		end
	end

	return args.input
end

function layer.vectorSubtract1D(args)
	for a = 1, #args.input do
		args.input[a] = args.input[a] - args.biases[a]
	end

	return args.input
end

function layer.vectorSubtract2D(args)
	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			args.input[a][b] = args.input[a][b] - args.biases[a][b]
		end
	end

	return args.input
end

function layer.vectorSubtract3D(args)
	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			for c = 1, #args.input[a][b] do
				args.input[a][b][c] = args.input[a][b][c] - args.biases[a][b][c]
			end
		end
	end

	return args.input
end

function layer.dot1D(args)
	for a = 1, #args.input do
		args.input[a] = args.input[a] * args.weights[a]
	end

	return args.input
end

function layer.dot2D(args)
	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			args.input[a][b] = args.input[a][b] * args.weights[a][b]
		end
	end

	return args.input
end

function layer.dot3D(args)
	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			for c = 1, #args.input[a][b] do
				args.input[a][b][c] = args.input[a][b][c] * args.weights[a][b][c]
			end
		end
	end

	return args.input
end

function layer.vectorDivide1D(args)
	for a = 1, #args.input do
		args.input[a] = args.input[a] / args.weights[a]
	end

	return args.input
end

function layer.vectorDivide2D(args)
	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			args.input[a][b] = args.input[a][b] / args.weights[a][b]
		end
	end

	return args.input
end

function layer.vectorDivide3D(args)
	for a = 1, #args.input do
		for b = 1, #args.input[a] do
			for c = 1, #args.input[a][b] do
				args.input[a][b][c] = args.input[a][b][c] / args.weights[a][b][c]
			end
		end
	end

	return args.input
end

function layer.dropOut(args)
	return args.input -- dropout is only applied during training, hence why were just returning the input
end

function layer.uniformNoise(args)
	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = layer.uniformNoise{
				input = args.input[a],
				lowerLimit = args.lowerLimit,
				upperLimit = args.upperLimit
			}
		else
			args.input[i] = args.input[a] + synmath.random.uniform(args.lowerLimit, args.upperLimit)
		end
	end

	return args.input
end

function layer.normalNoise(args)
	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = layer.normalNoise{
				input = args.input[a],
				mean = args.mean,
				sd = args.sd
			}
		else
			args.input[a] = args.input[a] + synmath.random.uniform(args.mean, args.sd)
		end
	end

	return args.input
end

function layer.softmax(args)
	return activation.softmax(args.input)
end

function layer.activate(args)
	local activation = activation[args.activation]

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = layer.activate{
				input = args.input[a],
				activation = args.activation,
				derivative = args.derivative,
				alpha = args.alpha
			}
		else
			args.input[a] = activation(args.input[a], args.derivative, args.alpha)
		end
	end

	return args.input
end

return layer
