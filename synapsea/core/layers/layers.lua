--[[
	https://github.com/mochji/synapsea
	core/layers/layers.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2023 mochji

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
]]--

local canindex          = require("core.utils.canindex")

local activationsModule = require("core.activations")

local layersModule = {
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
	upSample1D,
	upSample2D,
	upSample3D,
	zeroPad1D,
	zeroPad2D,
	zeroPad3D,
	crop1D,
	crop2D,
	crop3D,
	convolutional1D,
	convolutional2D,
	convolutional3D,
	convolutionalTranspose1D,
	convolutionalTranspose2D,
	convolutionalTranspose3D,
	flatten,
	reshape,
	add1D,
	add2D,
	add3D,
	subtract1D,
	subtract2D,
	subtract3D,
	multiply1D,
	multiply2D,
	multiply3D,
	divide1D,
	divide2D,
	divide3D,
	softmax,
	activate,
	dropout
}

function layersModule.dense(args)
	local activation = activationsModule[args.activation]
	local input, weights, bias, alpha, outputSize = args.input, args.weights, args.bias or 0, args.alpha, args.outputSize

	local output = {}

	for a = 1, outputSize do
		local sum = 0

		for b = 1, #args.input do
			sum = sum + input[b] * weights[b][a]
		end

		output[a] = activation(sum + bias, false, alpha)
	end

	return output
end

function layersModule.averagePooling1D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local outputSize = math.floor((#input - kernel[1]) / stride[1]) + 1
		local startIndex = 0

		local output = {}

		for a = 1, outputSize do
			local sum = 0

			for b = 1, #kernel[1] do
				if (startIndex + b) % dilation[1] == 0 then
					sum = sum + input[startIndex + b]
				end
			end

			startIndex = startIndex + stride[1]

			output[a] = sum / kernel[1]
		end

		return output
	end

	if canindex(args.input[1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.averagePooling2D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local kernelProduct = kernel[1] * kernel[2]
		local outputShape = {
			math.floor((#input - kernel[1]) / stride[1]) + 1,
			math.floor((#input[1] - kernel[2]) / stride[2]) + 1
		}
		local startIndexA = 0

		local output = {}

		for a = 1, outputShape[1] do
			local startIndexB = 0
			output[a] = {}

			for b = 1, outputShape[2] do
				local sum = 0

				for c = 1, kernel[1] do
					for d = 1, kernel[2] do
						if (startIndexA + c) % dilation[1] == 0 and (startIndexB + d) % dilation[2] == 0 then
							sum = sum + input[startIndexA + c][startIndexB + d]
						end
					end
				end

				startIndexB = startIndexB + stride[2]

				output[a][b] = sum / kernelProduct
			end

			startIndexA = startIndexA + stride[1]
		end

		return output
	end

	if canindex(args.input[1][1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.averagePooling3D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local kernelProduct = kernel[1] * kernel[2] * kernel[3]
		local outputShape = {
			math.floor((#input - kernel[1]) / stride[1]) + 1,
			math.floor((#input[1] - kernel[2]) / stride[2]) + 1,
			math.floor((#input[1][1] - kernel[3]) / stride[3]) + 1
		}
		local startIndexA = 0

		local output = {}

		for a = 1, outputShape[1] do
			output[a] = {}
			local startIndexB = 0

			for b = 1, outputShape[2] do
				output[a][b] = {}
				local startIndexC = 0

				for c = 1, outputShape[3] do
					local sum = 0

					for d = 1, kernel[1] do
						for e = 1, kernel[2] do
							for f = 1, kernel[3] do
								if (startIndexA + d) % dilation[1] == 0 and (startIndexB + e) % dilation[2] == 0 and (startIndex + f) % dilation[3] == 0 then
									sum = sum + input[startIndexA + d][startIndexB + e][startIndexC + f]
								end
							end
						end
					end

					startIndexC = startIndexC + stride[3]

					output[a][b][c] = sum / kernelProduct
				end

				startIndexB = startIndexB + stride[2]
			end

			startIndexA = startIndexA + stride[1]
		end

		return output
	end

	if canindex(args.input[1][1][1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.maxPooling1D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local outputSize = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1
		local startIndex = 0

		local output = {}

		for a = 1, outputSize do
			local max = input[startIndex + 1]

			for b = 1, #kernel[1] do
				if (startIndex + b) % dilation[1] == 0 then
					max = math.max(max, input[startIndex + b])
				end
			end

			start = start + stride[1]

			output[a] = max
		end

		return output
	end

	if canindex(args.input[1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.maxPooling2D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local outputShape = {
			math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1,
			math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1
		}
		local startIndexA = 0

		local output = {}

		for a = 1, outputShape[1] do
			output[a] = {}
			local startIndexB = 0

			for b = 1, outputShape[2] do
				local max = input[startIndexA + 1][startIndexB + 1]

				for c = 1, kernel[1] do
					for d = 1, kernel[2] do
						if (startIndexA + c) % dilation[1] == 0 and (startIndexB + d) % dilation[2] == 0 then
							max = math.max(max, input[startIndexA + c][startIndexB + d])
						end
					end
				end

				startIndexB = startIndexB + stride[2]

				output[a][b] = max
			end

			startIndexA = startIndexA + stride[1]
		end

		return output
	end

	if canindex(args.input[1][1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.maxPooling3D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local outputShape = {
			math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1,
			math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1,
			math.floor((#args.input[1][1] - args.kernel[3]) / args.stride[3]) + 1
		}
		local startIndexA = 0

		local output = {}

		for a = 1, outputShape[1] do
			output[a] = {}
			local startIndexB = 0

			for b = 1, outputShape[2] do
				output[a][b] = {}
				local startIndexC = 0

				for c = 1, outputShape[3] do
					local max = input[startIndexA + 1][startIndexB + 1][startIndexC + 1]

					for d = 1, kernel[1] do
						for e = 1, kernel[2] do
							for f = 1, kernel[3] do
								if (startIndexA + d) % dilation[1] == 0 and (startIndexB + e) % dilation[2] == 0 and (startIndexC + f) % dilation[3] == 0 then
									max = math.max(max, input[startIndexA + d][startIndexB + e][startIndexC + f])
								end
							end
						end
					end

					startIndexC = startIndexC + stride[3]

					output[a][b][c] = max
				end

				startIndexB = startIndexB + stride[2]
			end

			startIndexA = startIndexA + stride[1]
		end

		return output
	end

	if canindex(args.input[1][1][1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.sumPooling1D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local outputSize = math.floor((#input - kernel[1]) / stride[1]) + 1
		local startIndex = 0

		local output = {}

		for a = 1, outputSize do
			local sum = 0

			for b = 1, #kernel[1] do
				if (startIndex + b) % dilation[1] == 0 then
					sum = sum + input[startIndex + b]
				end
			end

			startIndex = startIndex + stride[1]

			output[a] = sum
		end

		return output
	end

	if canindex(args.input[1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.sumPooling2D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local outputShape = {
			math.floor((#input - kernel[1]) / stride[1]) + 1,
			math.floor((#input[1] - kernel[2]) / stride[2]) + 1
		}
		local startIndexA = 0

		local output = {}

		for a = 1, outputShape[1] do
			local startIndexB = 0
			output[a] = {}

			for b = 1, outputShape[2] do
				local sum = 0

				for c = 1, kernel[1] do
					for d = 1, kernel[2] do
						if (startIndexA + c) % dilation[1] == 0 and (startIndexB + d) % dilation[2] == 0 then
							sum = sum + input[startIndexA + c][startIndexB + d]
						end
					end
				end

				startIndexB = startIndexB + stride[2]

				output[a][b] = sum
			end

			startIndexA = startIndexA + stride[1]
		end

		return output
	end

	if canindex(args.input[1][1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.sumPooling3D(args)
	local function poolingFunc(input, kernel, stride, dilation)
		local outputShape = {
			math.floor((#input - kernel[1]) / stride[1]) + 1,
			math.floor((#input[1] - kernel[2]) / stride[2]) + 1,
			math.floor((#input[1][1] - kernel[3]) / stride[3]) + 1
		}
		local startIndexA = 0

		local output = {}

		for a = 1, outputShape[1] do
			output[a] = {}
			local startIndexB = 0

			for b = 1, outputShape[2] do
				output[a][b] = {}
				local startIndexC = 0

				for c = 1, outputShape[3] do
					local sum = 0

					for d = 1, kernel[1] do
						for e = 1, kernel[2] do
							for f = 1, kernel[3] do
								if (startIndexA + d) % dilation[1] == 0 and (startIndexB + e) % dilation[2] == 0 and (startIndexC + f) % dilation[3] == 0 then
									sum = sum + input[startIndexA + d][startIndexB + e][startIndexC + f]
								end
							end
						end
					end

					startIndexC = startIndexC + stride[3]

					output[a][b][c] = sum
				end

				startIndexB = startIndexB + stride[2]
			end

			startIndexA = startIndexA + stride[1]
		end

		return output
	end

	if canindex(args.input[1][1][1]) then
		local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

		local output = {}

		for a = 1, #input do
			for _, b in pairs(poolingFunc(input[a], kernel, stride, dilation)) do
				output[a] = b
			end
		end

		return output
	end

	return poolingFunc(args.input, args.kernel, args.stride, args.dilation)
end

function layersModule.upSample1D(args)
	local function upSampleFunc(input, kernel)
		local output = {}

		for a = 1, #input * kernel[1] do
			output[a] = input[math.ceil(a / kernel[1])]
		end

		return output
	end

	if canindex(args.input[1]) then
		local input, kernel = args.input, args.kernel

		local output = {}

		for a = 1, #input do
			for _, b in pairs(upSampleFunc(input[a], kernel)) do
				output[a] = b
			end
		end

		return output
	end

	return upSampleFunc(args.input, args.kernel)
end

function layersModule.upSample2D(args)
	local function upSampleFunc(input, kernel)
		local output = {}

		for a = 1, #input * kernel[1] do
			output[a] = {}

			for b = 1, #input[1] * kernel[2] do
				output[a][b] = input[math.ceil(a / kernel[1])][math.ceil(b / kernel[2])]
			end
		end

		return output
	end

	if canindex(args.input[1][1]) then
		local input, kernel = args.input, args.kernel

		local output = {}

		for a = 1, #input do
			for _, b in pairs(upSampleFunc(input, kernel)) do
				output[a] = b
			end
		end

		return output
	end

	return upSampleFunc(args.input, args.kernel)
end

function layersModule.upSample3D(args)
	local function upSampleFunc(input, kernel)
		local output = {}

		for a = 1, #input * kernel[1] do
			output[a] = {}

			for b = 1, #input * kernel[2] do
				output[a][b] = {}

				for c = 1, #input * kernel[3] do
					output[a][b][c] = input[math.ceil(a / kernel[1])][math.ceil(b / kernel[2])][math.ceil(c / kernel[3])]
				end
			end
		end

		return output
	end

	if canindex(args.input[1][1][1]) then
		local input, kernel = args.input, args.kernel

		local output = {}

		for a = 1, #input do
			for _, b in pairs(upSampleFunc(input, kernel)) do
				output[a] = b
			end
		end

		return output
	end

	return upSampleFunc(args.input, args.kernel)
end

function layersModule.zeroPad1D(args)
	local function zeroPadFunc(input, paddingAmount)
		for a = 1, paddingAmount[1] do
			table.insert(input, 1, 0)

			input[#input + 1] = 0
		end

		return input
	end

	if canindex(args.input[1]) then
		local input, paddingAmount = args.input, args.paddingAmount

		local output = {}

		for a = 1, #input do
			for _, b in pairs(zeroPadFunc(input, paddingAmount)) do
				output[a] = b
			end
		end
	
		return output
	end

	return zeroPadFunc(args.input, args.paddingAmount)
end

function layersModule.zeroPad2D(args)
	local function zeroPadFunc(input, paddingAmount)
		-- Pad the tops and bottoms

		for a = 1, paddingAmount[1] do
			table.insert(input, 1, {})

			for b = 1, #input[a + 1] do
				input[1][b] = 0
			end

			input[#input + 1] = input[1]
		end

		-- Pad the sides

		for a = 1, #input do
			for b = 1, paddingAmount[2] do
				table.insert(input[a], 1, 0)

				input[a][#input[a] + 1] = 0
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, paddingAmount = args.input, args.paddingAmount

		local output = {}

		for a = 1, #input do
			for _, b in pairs(zeroPadFunc(input, paddingAmount)) do
				output[a] = b
			end
		end

		return output
	end

	return zeroPadFunc(args.input, args.paddingAmount)
end

function layersModule.zeroPad3D(args)
	local function zeroPadFunc(input, paddingAmount)
		-- Pad the tops and bottoms

		for a = 1, paddingAmount[1] do
			table.insert(input, 1, {})

			for b = 1, #input[1] do
				for c = 1, #input[1][b] do
					input[1][b][c] = 0
				end
			end

			input[#input + 1] = {}

			for b = 1, #input[1] do
				input[#input[b]] = {}

				for c = 1, #input[1][b] do
					input[#input][b][c] = 0
				end
			end
		end

		-- Pad the sides

		for a = 1, #input do
			for b = 1, paddingAmount[2] do
				for c = 1, #input[1][1] do
					table.insert(input[a][1], 1, 0)
					input[a][#input[1] + 1][c] = 0
				end
			end
		end

		-- Pad the front and back

		for a = 1, #input do
			for b = 1, #input[1] do
				for c = 1, padingAmount[3] do
					table.insert(input[a][b], 1, 0)
					input[a][b][#input[1][1] + 1] = 0
				end
			end
		end

		return input
	end

	if canindex(args.input[1][1][1]) then
		local input, paddingAmount = args.input, args.paddingAmount

		local output = {}

		for a = 1, #input do
			for _, b in pairs(zeroPadFunc(input, paddingAmount)) do
				output[a] = b
			end
		end

		return output
	end

	return zeroPadFunc(input, paddingAmount)
end

function layersModule.crop1D(args)
	local function cropFunc(input, start, outputShape)
		local output = {}

		for a = start[1], outputShape[1] + start[1] do
			output[#output + 1] = input[a]
		end

		return output
	end

	if canindex(args.input[1]) then
		local input, start, outputShape = args.input, args.start, args.outputShape

		local output = {}

		for a = 1, #input do
			for _, b in pairs(cropFunc(input, start, outputShape)) do
				output[a] = b
			end
		end

		return output
	end

	return cropFunc(args.input, args.start, args.outputShape)
end

function layersModule.crop2D(args)
	local function cropFunc(input, start, outputShape)
		local output = {}

		for a = start[1], outputShape[1] + start[1] do
			local row = {}

			for b = start[2], outputShape[2] + start[2] do
				row[b] = input[a][b]
			end

			output[a] = row
		end

		return output
	end

	if canindex(args.input[1][1]) then
		local input, start, outputShape = args.input, args.start, args.outputShape

		local output = {}

		for a = 1, #input do
			for _, b in pairs(cropFunc(input, start, outputShape)) do
				output[a] = b
			end
		end

		return output
	end

	return cropFunc(args.input, args.start, args.outputShape)
end

function layersModule.crop3D(args)
	local function cropFunc(input, start, outputShape)
		local output = {}

		for a = start[1], outputShape[1] + start[1] do
			local crop = {}

			for b = start[2], outputShape[2] + start[2] do
				local row = {}

				for c = start[3], outputShape[3] + start[3] do
					row[c] = input[a][b][c]
				end

				crop[b] = row
			end

			output[a] = row
		end

		return output
	end

	if canindex(args.input[1][1]) then
		local input, start, outputShape = args.input, args.start, args.outputShape

		local output = {}

		for a = 1, #input do
			for _, b in pairs(cropFunc(input, start, outputShape)) do
				output[a] = b
			end
		end

		return output
	end

	return cropFunc(args.input, args.start, args.outputShape)
end

function layersModule.convolutional1D(args)
	local function convolutionalFunc(input, filter, stride, dilation, biases, alpha, activation)
		local outputSize = math.floor((#input - #filter[1]) / stride[1]) + 1

		local output = {}

		for a = 1, #filter do
			local startIndex = 0
			output[a] = {}

			for b = 1, outputSize do
				local sum = 0

				for c = 1, #filter[a] do
					if (startIndex + c) % dilation[1] == 0 then
						sum = sum + input[startIndex + c] * filter[a][c]
					end
				end

				startIndex = startIndex + stride[1]

				output[a][b] = sum
			end

			if biases then
				for b = 1, outputSize do
					output[a][b] = activation(output[a][b] + biases[b], false, alpha)
				end
			else
				for b = 1, outputSize do
					output[a][b] = activation(output[a][b], false, alpha)
				end
			end
		end

		if #output == 1 then
			return output[1]
		end

		return output
	end

	if canindex(args.input[1]) then
		local activation = activationsModule[args.activation]
		local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

		local output = {}

		for a = 1, #input do
			for _, b in pairs(convolutionalFunc(input[a], filter, stride, dilation, biases, alpha, activation)) do
				output[a] = b
			end
		end

		return output
	end

	return convolutionalFunc(args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, activationsModule[args.activation])
end

function layersModule.convolutional2D(args)
	local function convolutionalFunc(input, filter, stride, dilation, biases, alpha, activation)
		local outputShape = {
			math.floor((#input - #filter[1]) / stride[1]) + 1,
			math.floor((#input[1] - #filter[1][1]) / stride[2]) + 1
		}

		local output = {}

		for a = 1, #filter do
			local startIndexA = 0
			output[a] = {}

			for b = 1, outputShape[1] do
				local startIndexB = 0
				output[a][b] = {}

				for c = 1, outputShape[2] do
					local sum = 0

					for d = 1, #filter[a] do
						for e = 1, #filter[a][d] do
							if (startIndexA + d) % dilation[1] == 0 and (startIndexB + e) % dilation[2] == 0 then
								sum = sum + input[startIndexA + d][startIndexB + e] * filter[a][d][e]
							end
						end
					end

					startIndexB = startIndexB + stride[2]

					output[a][b][c] = sum
				end

				startIndexA = startIndexA + 1
			end

			if biases then
				for b = 1, outputShape[1] do
					for c = 1, outputShape[2] do
						output[a][b][c] = activation(output[a][b][c] + biases[b][c], false, alpha)
					end
				end
			else
				for b = 1, outputShape[1] do
					for c = 1, outputShape[2] do
						output[a][b][c] = activation(output[a][b][c], false, alpha)
					end
				end
			end
		end

		if #output == 1 then
			return output[1]
		end

		return output
	end

	if canindex(args.input[1][1]) then
		local activation = activationsModule[args.activation]
		local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

		local output = {}

		for a = 1, #input do
			for _, b in pairs(convolutionalFunc(input[a], filter, stride, dilation, biases, alpha, activation)) do
				output[a] = b
			end
		end

		return output
	end

	return convolutionalFunc(args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, activationsModule[args.activation])
end

function layersModule.convolutional3D(args)
	local function convolutionalFunc(input, filter, stride, dilation, biases, alpha, activation)
		local outputShape = {
			math.floor((#input - #filter[1]) / stride[1]) + 1,
			math.floor((#input[1] - #filter[1][1]) / stride[2]) + 1,
			math.floor((#input - #filter[1][1][1]) / stride[3]) + 1
		}

		local output = {}

		for a = 1, #filter do
			local startIndexA = 0
			output[a] = {}

			for b = 1, outputShape[1] do
				local startIndexB = 0
				output[a][b] = {}

				for c = 1, outputShape[2] do
					local startIndexC = 0
					output[a][b][c] = {}

					for d = 1, outputShape[3] do
						local sum = 0

						for e = 1, #filter[a] do
							for f = 1, #filter[a][e] do
								for g = 1, #filter[a][e][f] do
									if (startIndexA + e) % dilation[1] == 0 and (startIndexB + f) % dilation[2] == 0 and (startIndexC + g) % dilation[3] == 0 then
										sum = sum + input[startIndexA + e][startIndexB + f][startIndexC + g] * filter[a][e][f][g]
									end
								end
							end
						end

						startIndexC = startIndexC + stride[3]

						output[a][b][c][d] = sum
					end

					startIndexB = startIndexB + 1
				end

				startIndexA = startIndexA + 1
			end

			if biases then
				for b = 1, outputShape[1] do
					for c = 1, outputShape[2] do
						for d = 1, outputShape[3] do
							output[a][b][c][d] = activation(output[a][b][c][d] + biases[b][c][d], false, alpha)
						end
					end
				end
			else
				for b = 1, outputShape[1] do
					for c = 1, outputShape[2] do
						for d = 1, outputShape[3] do
							output[a][b][c][d] = activation(output[a][b][c][d], false, alpha)
						end
					end
				end
			end
		end

		if #output == 1 then
			return output[1]
		end

		return output
	end

	if canindex(args.input[1][1][1]) then
		local activation = activationsModule[args.activation]
		local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

		local output = {}
	
		for a = 1, #input do
			for _, b in pairs(convolutionalFunc(input[a], filter, stride, dilation, biases, alpha, activation)) do
				output[a] = b
			end
		end

		return output
	end

	return convolutionalFunc(args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, activationsModule[args.activation])
end

function layersModule.convolutionalTranspose1D(args)
	local function convolutionalTransposeFunc(input, filter, stride, dilation, biases, alpha, activation, paddingAmount)
		return layersModule.convolutional1D{
			input = layersModule.zeroPad1D{
				input = input,
				paddingAmount = paddingAmount
			},
			filter = filter,
			stride = stride,
			dilation = dilation,
			biases = biases,
			alpha = alpha,
			activation = activation
		}
	end

	if canindex(args.input[1]) then
		local input, filter, stride, dilation, biases, alpha, activation, paddingAmount = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, args.activation, args.paddingAmount
		local activation = activationsModule[args.activation]

		local output = {}

		for a = 1, #input do
			for _, b in pairs(convolutionalTransposeFunc(input[a], filter, stride, dilation, biases, alpha, activation, paddingAmount)) do
				output[a] = b
			end
		end

		return output
	end

	return convolutionalTransposeFunc(args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, activationsModule[args.activation], args.paddingAmount)
end

function layersModule.convolutionalTranspose2D(args)
	local function convolutionalTransposeFunc(input, filter, stride, dilation, biases, alpha, activation, paddingAmount)
		return layersModule.convolutional2D{
			input = layersModule.zeroPad2D{
				input = input,
				paddingAmount = paddingAmount
			},
			filter = filter,
			stride = stride,
			dilation = dilation,
			biases = biases,
			alpha = alpha,
			activation = activation
		}
	end

	if canindex(args.input[1][1]) then
		local input, filter, stride, dilation, biases, alpha, activation, paddingAmount = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, args.activation, args.paddingAmount
		local activation = activationsModule[args.activation]

		local output = {}

		for a = 1, #input do
			for _, b in pairs(convolutionalTransposeFunc(input[a], filter, stride, dilation, biases, alpha, activation, paddingAmount)) do
				output[a] = b
			end
		end

		return output
	end

	return convolutionalTransposeFunc(args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, activationsModule[args.activation], args.paddingAmount)
end

function layersModule.convolutionalTranspose3D(args)
	local function convolutionalTransposeFunc(input, filter, stride, dilation, biases, alpha, activation, paddingAmount)
		return layersModule.convolutional3D{
			input = layersModule.zeroPad3D{
				input = input,
				paddingAmount = paddingAmount
			},
			filter = filter,
			stride = stride,
			dilation = dilation,
			biases = biases,
			alpha = alpha,
			activation = activation
		}
	end

	if canindex(args.input[1][1][1]) then
		local input, filter, stride, dilation, biases, alpha, activation, paddingAmount = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, args.activation, args.paddingAmount
		local activation = activationsModule[args.activation]

		local output = {}

		for a = 1, #input do
			for _, b in pairs(convolutionalTransposeFunc(input[a], filter, stride, dilation, biases, alpha, activation, paddingAmount)) do
				output[a] = b
			end
		end

		return output
	end

	return convolutionalTransposeFunc(args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha, activationsModule[args.activation], args.paddingAmount)
end

function layersModule.flatten(args)
	local flattenFunc

	flattenFunc = function(input)
		local output = {}

		for a = 1, #input do
			if canindex(input[a]) then
				local flattened = flattenFunc(input[a])

				for b = 1, #flattened do
					output[#output + 1] = flattened[b]
				end
			else
				output[#output + 1] = input[a]
			end
		end

		return output
	end

	return flattenFunc(args.input)
end

function layersModule.reshape(args)
	local flattenFunc, reshapeFunc

	flattenFunc = function(input)
		local output = {}

		for a = 1, #input do
			if canindex(input[a]) then
				local flattened = flattenFunc(input[a])

				for b = 1, #flattened do
					output[#output + 1] = flattened[b]
				end
			else
				output[#output + 1] = input[a]
			end
		end

		return output
	end

	reshapeFunc = function(input, shape, index)
		local output = {}

		local inputIndex = 0

		if shape[index + 1] then
			for b = 1, #shape[index] do
				output[a] = reshapeFunc(input, shape, index + 1)
			end
		else
			for a = 1, shape[index] do
				output[a]  = input[inputIndex]
				inputIndex = inputIndex + 1
			end
		end
	end

	return reshapeFunc(flattenFunc(args.input), args.shape, 1)
end

function layersModule.add1D(args)
	local function addFunc(input, biases)
		for a = 1, #input do
			input[a] = input[a] + biases[a]
		end

		return input
	end

	if canindex(args.input[1]) then
		local input, biases = args.input, args.biases

		local output = {}

		for a = 1, #input do
			for _, b in pairs(addFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return addFunc(args.input, args.biases)
end

function layersModule.add2D(args)
	local function addFunc(input, biases)
		for a = 1, #input do
			for b = 1, #input[a] do
				input[a][b] = input[a][b] + biases[a][b]
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, biases = args.input, args.biases

		local output = {}

		for a = 1, #input do
			for _,b in pairs(addFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return addFunc(args.input, args.biases)
end

function layersModule.add3D(args)
	local function addFunc(input, biases)
		for a = 1, #input do
			for b = 1, #input[a] do
				for c = 1, #input[a][b] do
					input[a][b][c] = input[a][b][c] + biases[a][b][c]
				end
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, biases = args.input, args.biases

		local output = {}

		for a = 1, #input do
			for _, b in pairs(addFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return addFunc(args.input, args.biases)
end

function layersModule.subtract1D(args)
	local function subtractFunc(input, biases)
		for a = 1, #input do
			input[a] = input[a] - biases[a]
		end

		return input
	end

	if canindex(args.input[1]) then
		local input, biases = args.input, args.biases

		local output = {}

		for a = 1, #input do
			for _, b in pairs(subtractFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return subtractFunc(args.input, args.biases)
end

function layersModule.subtract2D(args)
	local function subtractFunc(input, biases)
		for a = 1, #input do
			for b = 1, #input[a] do
				input[a][b] = input[a][b] - biases[a][b]
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, biases = args.input, args.biases

		local output = {}

		for a = 1, #input do
			for _, b in pairs(subtractFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return subtractFunc(args.input, args.biases)
end

function layersModule.subtract3D(args)
	local function subtractFunc(input, biases)
		for a = 1, #input do
			for b = 1, #input[a] do
				for c = 1, #input[a][b] do
					input[a][b][c] = input[a][b][c] - biases[a][b][c]
				end
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, biases = args.input, args.biases

		local output = {}

		for a = 1, #input do
			for _, b in pairs(subtractFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return subtractFunc(args.input, args.biases)
end

function layersModule.multiply1D(args)
	local function multiplyFunc(input, weights)
		for a = 1, #input do
			input[a] = input[a] * weights[a]
		end

		return input
	end

	if canindex(args.input[1]) then
		local input, weights = args.input, args.weights

		local output = {}

		for a = 1, #input do
			for _, b in pairs(multiplyFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return multiplyFunc(args.input, args.weights)
end

function layersModule.multiply2D(args)
	local function multiplyFunc(input, weights)
		for a = 1, #input do
			for b = 1, #input[a] do
				input[a][b] = input[a][b] * weights[a][b]
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, weights = args.input, args.weights

		local output = {}

		for a = 1, #input do
			for _, b in pairs(multiplyFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return multiplyFunc(args.input, args.weights)
end

function layersModule.multiply3D(args)
	local function multiplyFunc(input, weights)
		for a = 1, #input do
			for b = 1, #input[a] do
				for c = 1, #input[a][b] do
					input[a][b][c] = input[a][b][c] * weights[a][b][c]
				end
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, weights = args.input, args.weights

		local output = {}

		for a = 1, #input do
			for _, b in pairs(multiplyFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return multiplyFunc(args.input, args.weights)
end

function layersModule.divide1D(args)
	local function divideFunc(input, weights)
		for a = 1, #input do
			input[a] = input[a] / weights[a]
		end

		return input
	end

	if canindex(args.input[1]) then
		local input, weights = args.input, args.weights

		local output = {}

		for a = 1, #input do
			for _, b in pairs(divideFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return divideFunc(args.input, args.weights)
end

function layersModule.divide2D(args)
	local function divideFunc(input, weights)
		for a = 1, #input do
			for b = 1, #input[a] do
				input[a][b] = input[a][b] / weights[a][b]
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, weights = args.input, args.weights

		local output = {}

		for a = 1, #input do
			for _, b in pairs(divideFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return divideFunc(args.input, args.weights)
end

function layersModule.divide3D(args)
	local function divideFunc(input, weights)
		for a = 1, #input do
			for b = 1, #input[a] do
				for c = 1, #input[a][b] do
					input[a][b][c] = input[a][b][c] / weights[a][b][c]
				end
			end
		end

		return input
	end

	if canindex(args.input[1][1]) then
		local input, weights = args.input, args.weights

		local output = {}

		for a = 1, #input do
			for _, b in pairs(divideFunc(input[a], biases)) do
				output[a] = b
			end
		end

		return output
	end

	return divideFunc(args.input, args.weights)
end

function layersModule.softmax(args)
	return activationsModule.softmax(args.input)
end

function layersModule.activate(args)
	local activateFunc

	activateFunc = function(input, activation, derivative, alpha)
		for a = 1, #input do
			if canindex(input[a]) then
				input[a] = activateFunc(
					input,
					activation,
					derivative,
					alpha
				)
			else
				input[a] = activation(input[a], derivative, alpha)
			end
		end

		return input
	end

	return activationFunc(args.input, activationsModule[args.activation], args.derivative, args.alpha)
end

function layersModule.dropout(args)
	return args.input
end

return layersModule
