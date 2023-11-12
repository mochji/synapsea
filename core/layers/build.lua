--[[
	https://github.com/mochji/synapsea
	core/layers/build.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
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

local buildModule = {
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
	convolutional1D,
	convolutional2D,
	convolutional3D,
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
	dropOut
}

function buildModule.dense(args)
	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			outputSize = args.outputSize
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {args.outputSize}
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = args.weightsInitializer or "constant",
		parameters = args.weightsInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.bias = {
			initializer = args.biasInitializer or "constant",
			parameters = args.biasInitParameters or {value = 0}
		}
	end

	-- Trainable

	if args.weightsTrainable then
		layer.trainable.weights = true
	end

	if args.biasTrainable and args.useBias then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.weights = {
		shape = {args.inputShape[1], args.outputSize}
	}

	if args.useBias then
		layer.parameters.bias = 0
	end

	return layer, parameterBuild
end

function buildModule.averagePooling1D(args)
	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride or {1},
			dilation = args.dilation or {1}
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1}
	}

	return layer
end

function buildModule.averagePooling2D(args)
	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride or {1, 1},
			dilation = args.dilation or {1, 1}
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1}
	}

	return layer
end

function buildModule.averagePooling3D(args)
	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride or {1, 1, 1},
			dilation = args.dilation or {1, 1, 1}
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1}
	}

	return layer
end

buildModule.maxPooling1D = buildModule.averagePooling1D

buildModule.maxPooling2D = buildModule.averagePooling2D

buildModule.maxPooling3D = buildModule.averagePooling3D

buildModule.sumPooling1D = buildModule.averagePooling1D

buildModule.sumPooling2D = buildModule.averagePooling2D

buildModule.sumPooling3D = buildModule.averagePooling3D

function buildModule.averageGlobalPooling1D(args)
	return {
		inputShape = args.inputShape,
		outputShape = {1}
	}
end

function buildModule.averageGlobalPooling2D(args)
	return {
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1]}
	}
end

buildModule.averageGlobalPooling3D = buildModule.averageGlobalPooling2D

buildModule.maxGlobalPooling1D = buildModule.averageGlobalPooling1D

buildModule.maxGlobalPooling2D = buildModule.averageGlobalPooling2D

buildModule.maxGlobalPooling3D = buildModule.averageGlobalPooling3D

buildModule.sumGlobalPooling1D = buildModule.averageGlobalPooling1D

buildModule.sumGlobalPooling2D = buildModule.averageGlobalPooling2D

buildModule.sumGlobalPooling3D = buildModule.averageGlobalPooling3D

function buildModule.upSample1D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1]}
	}
end

function buildModule.upSample2D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2]}
	}
end

function buildModule.upSample3D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2], args.inputShape[3] * args.kernel[3]}
	}
end

function buildModule.zeroPad1D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2}
	}
end

function buildModule.zeroPad2D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2}
	}
end

function buildModule.zeroPad3D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2, args.inputShape[3] + args.paddingAmount[3] * 2}
	}
end

function buildModule.crop1D(args)
	return {
		config = {
			start = args.start,
			outputShape = args.outputShape
		},
		inputShape = args.inputShape,
		outputShape = args.outputShape
	}
end

buildModule.crop2D = buildModule.crop1D

buildModule.crop3D = buildModule.crop1D

function buildModule.convolutional1D(args)
	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride or {1},
			dilation = args.dilation or {1},
			filters = args.filters or 1
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1}
	}, {}

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInitializer or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer or "constant",
			parameters = args.biasesInitParameters or {value = 0}
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function buildModule.convolutional2D(args)
	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride or {1, 1},
			dilation = args.dilation or {1, 1},
			filters = args.filters or 1
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1}
	}, {}

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInitializer or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer or "constant",
			parameters = args.biasesInitParameters or {value = 0}
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1], args.kernel[2]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function buildModule.convolutional3D(args)
	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride or {1, 1, 1},
			dilation = args.dilation or {1, 1, 1},
			filters = args.filters or 1
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1}
	}, {}

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInitializer or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer or "constant",
			parameters = args.biasesInitParameters or {value = 0}
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1], args.kernel[2], args.kernel[3]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function buildModule.flatten(args)
	local outputShape = 1

	for a = 1, #args.inputShape do
		outputShape = outputShape * args.inputShape[a]
	end

	return {
		inputShape = args.inputShape,
		outputShape = {outputShape}
	}
end

function buildModule.add1D(args)
	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = args.biasesInitializer or "constant",
		parameters = args.biasesInitParameters or {value = 0}
	}

	-- Trainable

	if args.biasTrainable then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.biases = {
		shape = args.inputShape
	}

	return layer, parameterBuild
end

buildModule.add2D = buildModule.add1D

buildModule.add3D = buildModule.add1D

buildModule.subtract1D = buildModule.add1D

buildModule.subtract2D = buildModule.add1D

buildModule.subtract3D = buildModule.add1D

function buildModule.multiply1D(args)
	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = args.weightsInitializer or "constant",
		parameters = args.weightsInitParameters or {value = 0}
	}

	-- Trainable

	if args.biasTrainable then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.weights = {
		shape = args.inputShape
	}

	return layer, parameterBuild
end

buildModule.multiply2D = buildModule.multiply1D

buildModule.multiply3D = buildModule.multiply1D

buildModule.vectorDivide1D = buildModule.multiply1D

buildModule.divide2D = buildModule.multiply1D

buildModule.divide3D = buildModule.multiply1D

function buildModule.softmax(args)
	return {
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

function buildModule.activate(args)
	return {
		config = {
			activation = args.activation,
			derivative = args.derivative
		},
		parameters = {
			alpha = args.alpha
		},
		inputShape = args.inputShape,
		outputShape = args.outputShape
	}
end


function buildModule.dropOut(args)
	return {
		config = {
			rate = args.rate
		},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

function buildModule.uniformNoise(args)
	return {
		config = {
			lowerLimit = args.lowerLimit,
			upperLimit = args.upperLimit
		},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

function buildModule.normalNoise(args)
	return {
		config = {
			mean = args.mean,
			sd = args.sd,
		},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

return buildModule
